# -------------------------------------------------------------------------
#
# FuriosaAI CONFIDENTIAL
# __________________
#
# FuriosaAI Incorporated, All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of FuriosaAI and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to FuriosaAI Inc
# and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from FuriosaAI Inc.
# --------------------------------------------------------------------------
from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

import onnx
import os
import tqdm

import numpy as np
import onnxruntime as ort

from onnx import TensorProto, ModelProto
from onnx.helper import make_node, make_tensor_value_info

from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model
from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import __DYNAMIC_RANGE_COLLECTORS__, get_input_tensors, get_vi_dtype

__ALL_COLLECTORS__ = sum(__DYNAMIC_RANGE_COLLECTORS__.values(), [])


class ONNXCalibrator:
    def __init__(self, model: onnx.ModelProto):
        copy_model = ModelProto()
        copy_model.CopyFrom(model)
        self.model = copy_model

        self.input_tensors = get_input_tensors(model)
        self.value_info = {vi.name: vi for vi in
                           list(self.model.graph.value_info) + list(self.model.graph.input) +
                           list(self.model.graph.output)}

        # raise Exception if input_tensor is not defined in model.graph.input
        for input_tensor in self.input_tensors:
            if input_tensor not in [input.name for input in self.model.graph.input]:
                raise Exception('input_tensor: %s is not defined in model.graph.input' % input_tensor)

    def build_calibration_model(self) -> onnx.ModelProto:
        return self.augment_model()

    def calibrate(
        self, dataset: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Tuple[float, float]]:
        """Estimates the range of tensors, based on a dataset.

        Args:
            dataset: A calibration dataset.

        Returns:
            A dict mapping tensors that precede ReduceMin and ReduceMax
            to their minimum and maximum values.
        """
        return calibrate(self.model, dataset)

    def calibrate_with_random(self, num_data: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
        '''
        Gather intermediate model outputs after running inference
            parameter model_path: path to augmented FP32 ONNX model
            parameter inputs: list of loaded test inputs (or image matrices)
            return: dictionary mapping added node names to (ReduceMin, ReduceMax) pairs
        '''
        # Log severity level 3(Error) in order not to print warnings(level 2)
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(self.model.SerializeToString(), None)

        input_names = [attr.name for attr in sess.get_inputs()]
        input_shapes = [attr.shape for attr in sess.get_inputs()]
        input_types = [attr.type for attr in sess.get_inputs()]

        feed_dict = dict()

        calibration_dataset = []
        for _ in range(num_data or 10):
            for (name, shape, type) in zip(input_names, input_shapes, input_types):
                if type == 'tensor(float)':
                    dtype = np.float32
                elif type == 'tensor(int64)':
                    dtype = np.int64
                else:
                    raise Exception('Unknown dtype: %s' % type)
                batch_size = 1
                feed_dict[name] = np.random.random((batch_size, *shape[1:])).astype(dtype)
            calibration_dataset.append(feed_dict)

        observers = [output.name for output in sess.get_outputs() if
                     'ReduceMin' in output.name or 'ReduceMax' in output.name]

        disabled = True if os.environ.get('TQDM_DISABLE') else False
        observed_vals = [sess.run(observers, feed_dict)
                         for feed_dict in tqdm.tqdm(calibration_dataset, desc='Calibration', disable=disabled)]

        val_dicts = dict(zip(observers, zip(*observed_vals)))

        node_names = [key.rpartition('_')[0] for key in val_dicts.keys() if 'ReduceMax' in key]
        min_dicts = [float(np.min(value)) for key, value in val_dicts.items() if 'ReduceMin' in key]
        max_dicts = [float(np.max(value)) for key, value in val_dicts.items() if 'ReduceMax' in key]

        return dict(zip(node_names, zip(min_dicts, max_dicts)))

    def augment_model(self):
        new_list = []

        for input in self.input_tensors:
            dtype = get_vi_dtype(self.value_info[input])

            if dtype != onnx.TensorProto.FLOAT:
                continue

            new_list += self._attach_minmax_observer(input)

        for node in self.model.graph.node:
            new_list.append(node)
            for output in node.output:
                dtype = get_vi_dtype(self.value_info[output])

                if dtype != onnx.TensorProto.FLOAT:
                    continue

                new_list += self._attach_minmax_observer(output)

        self.model = utils.rebuild_model(self.model, new_list)
        check_model(self.model)

        return self.model

    def _attach_minmax_observer(self, node_output_name):
        # Adding ReduceMin nodes
        reduce_min_name = node_output_name + '_ReduceMin'
        reduce_min_node = make_node('ReduceMin', [node_output_name],
                                    [reduce_min_name], reduce_min_name, keepdims=0)
        reduce_min_info = make_tensor_value_info(reduce_min_node.output[0], TensorProto.FLOAT, ())

        # Adding ReduceMax nodes
        reduce_max_name = node_output_name + '_ReduceMax'
        reduce_max_node = make_node('ReduceMax', [node_output_name],
                                    [reduce_max_name], reduce_max_name, keepdims=0)
        reduce_max_info = make_tensor_value_info(reduce_max_node.output[0], TensorProto.FLOAT, ())

        self.model.graph.output.extend([reduce_min_info, reduce_max_info])
        return reduce_min_node, reduce_max_node


def calibrate(
    model: ModelProto, dataset: List[Dict[str, np.ndarray]]
) -> Dict[str, Tuple[float, float]]:
    """Estimates the range of tensors in a model, based on a dataset.

    Args:
        model: An ONNX model augmented with ReduceMin and ReduceMax.
        dataset: A calibration dataset.

    Returns:
        A dict mapping tensors that precede ReduceMin and ReduceMax to
        their minimum and maximum values.
    """
    ort.set_default_logger_severity(3)
    session = ort.InferenceSession(model.SerializeToString())

    reduces = [
        output.name
        for output in session.get_outputs()
        if (output.name.endswith("_ReduceMin") or output.name.endswith("_ReduceMax"))
    ]

    minimum = defaultdict(lambda: math.inf)
    maximum = defaultdict(lambda: -math.inf)
    if not os.environ.get("TQDM_DISABLE"):
        dataset = tqdm.tqdm(dataset, desc="Calibration")
    for inputs in dataset:
        reduce_vals = session.run(reduces, inputs)
        for reduce, reduce_val in zip(reduces, reduce_vals):
            if reduce.endswith("_ReduceMin"):
                name = reduce[: reduce.rfind("_ReduceMin")]
                if minimum[name] > reduce_val:
                    minimum[name] = reduce_val
            elif reduce.endswith("_ReduceMax"):
                name = reduce[: reduce.rfind("_ReduceMax")]
                if maximum[name] < reduce_val:
                    maximum[name] = reduce_val
    return {name: (minimum[name], maximum[name]) for name in minimum}
