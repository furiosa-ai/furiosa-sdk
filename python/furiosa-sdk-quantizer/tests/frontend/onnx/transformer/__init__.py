import abc
from typing import List, Tuple, Union
import unittest

import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np

from tests import torch_to_onnx
from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer.polish_model import PolishModel


class TestTransformer(unittest.TestCase):
    @staticmethod
    def make_test_model(torch_model: callable, transformer: Union[Transformer, abc.ABCMeta],
                        input_shapes: List[Tuple[int, ...]]):
        orig_model = torch_to_onnx(torch_model, input_shapes)
        # apply polish_model by default
        orig_model = PolishModel().transform(orig_model)
        copy_model = onnx.ModelProto()
        copy_model.CopyFrom(orig_model)

        if isinstance(transformer, Transformer):
            trans_model = transformer.transform(copy_model)
        elif isinstance(transformer, abc.ABCMeta):
            trans_model = transformer(copy_model).transform()
        else:
            raise Exception()

        return orig_model, trans_model

    @staticmethod
    def make_test_unit_model_from_onnx(onnx_model: onnx.ModelProto, transformer: Transformer):
        copy_model = onnx.ModelProto()

        copy_model.CopyFrom(onnx_model)
        trans_model = transformer.transform(copy_model)

        return onnx_model, trans_model

    def check_graph_node(self, model, op_types: List[str]):
        assert len(model.graph.node) == len(op_types)
        for node, op_type in zip(model.graph.node, op_types):
            self.assertTrue(node.op_type == op_type)

    def check_output_value(self, orig_model, trans_model, input_shapes):
        np_arrays = [np.random.randn(*shape).astype(np.float32) for shape in input_shapes]
        actual = run_onnx_model(orig_model, np_arrays)
        expected = run_onnx_model(trans_model, np_arrays)

        for act, exp in zip(actual, expected):
            self.assertListAlmostEqual(act, exp, 4, msg=f"{np_arrays}")

    def check_value_info(self, model):
        value_info = {vi.name: vi for vi in
                      list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}

        for node in model.graph.node:
            for input in list(node.input) + list(node.output):
                if not input:
                    continue
                self.assertTrue(input in value_info.keys())

    def check_initializer(self, actual, expected):
        self.assertListAlmostEqual(actual.flatten().tolist(), expected.flatten().tolist(), 4)

    def check_attribute(self, actual, expected):
        self.assertEqual(actual, expected)

    def check_assertion(self, func, kwargs):
        self.assertRaises(AssertionError, func, **kwargs)

    def assertListAlmostEqual(self, list1, list2, tol, msg=None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol, msg=msg)


def run_onnx_model(model: onnx.ModelProto,
                   input_arrays: List[np.array]) -> List[List[float]]:
    sess = ort.InferenceSession(model.SerializeToString())
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    feed_dict = {k: v for k, v in zip(input_names, input_arrays)}
    outputs = sess.run(output_names, input_feed=feed_dict)

    flattened_outputs = [val.flatten().tolist() for val in outputs]

    return flattened_outputs


def init_to_numpy(model, init_name):
    initializer = {init.name: init for init in model.graph.initializer}
    return numpy_helper.to_array(initializer[init_name])
