#!/usr/bin/env python3
from pathlib import Path
import pickle
import unittest

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx import (
    calibrate,
    optimize_model,
    post_training_quantization_with_random_calibration,
    post_training_quantize,
    quantizer,
)
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model


class ONNXTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # val2017-10.pickle contains a calibration dataset that consists
        # of 10 images in the COCO validation dataset [val2017.zip][].
        #
        # [val2017.zip]: http://images.cocodataset.org/zips/val2017.zip
        with open(Path(__file__).resolve().parent / "val2017-10.pickle", "rb") as f:
            cls.dataset = pickle.load(f)

        # load efficientdet_d0
        cls.effdet = onnx.load(Path(__file__).resolve().parent / "efficientdet_d0-f3276ba8.onnx")
        cls.opt_effdet = optimize_model(cls.effdet)
        cls.ranges = calibrate.calibrate(cls.opt_effdet, cls.dataset)

    def test_post_training_quantize_with_raw_data_True(self):
        quant_model = post_training_quantize(self.effdet, self.dataset, False)
        self._ensure_no_initializer_in_graph_input(quant_model)

    def test_post_training_quantize_with_raw_data_False(self):
        quant_model = quantizer.FuriosaONNXQuantizer(
            model=self.opt_effdet,
            per_channel=False,
            static=True,
            mode=quantizer.QuantizationMode.DFG,
            dynamic_ranges=self.ranges,
            raw_data=False,
        ).quantize()
        self._ensure_no_initializer_in_graph_input(quant_model)

    def test_post_training_quantize_with_FAKE_mode(self):
        quant_model = quantizer.FuriosaONNXQuantizer(
            model=self.opt_effdet,
            per_channel=True,
            static=True,
            mode=quantizer.QuantizationMode.FAKE,
            dynamic_ranges=self.ranges,
            raw_data=True,
        ).quantize()
        self._ensure_no_initializer_in_graph_input(quant_model)
        self._ensure_no_unused_initializer(quant_model)

    def test_zero_bias_scale(self):  # pylint: disable=no-self-use
        model = _make_zero_bias_scale_model()
        model = post_training_quantization_with_random_calibration(
            model, per_channel=True, static=True, mode=QuantizationMode.DFG
        )
        self._ensure_no_initializer_in_graph_input(model)

    def _ensure_no_initializer_in_graph_input(self, model: onnx.ModelProto) -> None:
        graph_inputs = set(value_info.name for value_info in model.graph.input)
        self.assertTrue(all(tensor.name not in graph_inputs for tensor in model.graph.initializer))

    def _ensure_no_unused_initializer(self, model: onnx.ModelProto) -> None:
        node_input_names = set(
            tensor_name for node in model.graph.node for tensor_name in node.input
        )
        self.assertTrue(all(tensor.name in node_input_names for tensor in model.graph.initializer))

    def _ensure_no_unused_initializer(self, model: onnx.ModelProto) -> None:
        node_input_names = set(
            tensor_name for node in model.graph.node for tensor_name in node.input
        )
        self.assertTrue(all(tensor.name in node_input_names for tensor in model.graph.initializer))


def _make_zero_bias_scale_model():
    input_shape = [1, 1, 3, 3]
    output_shape = [1, 1, 1, 1]
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": np.array(
                    [
                        [
                            [
                                [
                                    -2.802596928649634e-43,
                                    -6.165713243029195e-44,
                                    -3.825544807606751e-43,
                                ],
                                [
                                    8.828180325246348e-44,
                                    2.059908742557481e-43,
                                    4.624284932271896e-44,
                                ],
                                [
                                    6.866362475191604e-44,
                                    2.382207389352189e-44,
                                    6.726232628759122e-44,
                                ],
                            ]
                        ]
                    ],
                    dtype=np.float32,
                ),
                "b": np.array([-4.312656879425049], dtype=np.float32),
            },
            "node": [("Conv", ["x", "w", "b"], "y")],
        }
    )


if __name__ == "__main__":
    unittest.main()
