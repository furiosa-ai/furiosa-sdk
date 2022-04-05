#!/usr/bin/env python3
from pathlib import Path
import pickle
import unittest

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx import (
    _is_fully_quantized,
    _is_fully_quantized_in_dfg_mode,
    _is_fully_quantized_in_fake_quant_mode,
    _is_sandwiched,
    calibrate,
    optimize_model,
    parse_onnx_graph,
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

    def test_post_training_quantize_with_raw_data_True(self):
        post_training_quantize(self.effdet, self.dataset, False)

    def test_post_training_quantize_with_raw_data_False(self):
        model = optimize_model(self.effdet)
        ranges = calibrate.calibrate(model, self.dataset)
        quantizer.FuriosaONNXQuantizer(
            model=model,
            per_channel=False,
            static=True,
            mode=quantizer.QuantizationMode.DFG,
            dynamic_ranges=ranges,
            raw_data=False,
        ).quantize()

    def test__is_sandwiched(self):
        model = _make_sandwiced_model()
        test_node = model.graph.node[2]
        self.assertTrue(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched2(self):
        model = _make_sandwiced_i64_model()
        test_node = model.graph.node[0]
        self.assertTrue(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched3(self):
        model = _make_partially_sandwiched_top_model()
        test_node = model.graph.node[2]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched4(self):
        model = _make_partially_sandwiced_model()
        test_node = model.graph.node[1]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched5(self):
        model = _make_partially_sandwiced_bot_model()
        test_node = model.graph.node[0]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode(self):
        model = _make_sandwiced_model()
        self.assertTrue(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode2(self):
        model = _make_dfg_quantized_model()
        self.assertTrue(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode3(self):
        model = _make_partially_sandwiched_top_model()
        self.assertFalse(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_fake_quant_mode(self):
        model = _make_sandwiced_model()
        self.assertTrue(
            _is_fully_quantized_in_fake_quant_mode(model.graph, *parse_onnx_graph(model))
        )

    def test__is_fully_quantized_in_fake_quant_mode2(self):
        model = _make_partially_sandwiced_bot_model()
        self.assertFalse(
            _is_fully_quantized_in_fake_quant_mode(model.graph, *parse_onnx_graph(model))
        )

    def test__is_fully_quantized(self):
        model = _make_dfg_quantized_model()
        self.assertTrue(_is_fully_quantized(model))

    def test_zero_bias_scale(self):  # pylint: disable=no-self-use
        model = _make_zero_bias_scale_model()
        model = post_training_quantization_with_random_calibration(
            model, per_channel=True, static=True, mode=QuantizationMode.DFG
        )


def _make_dfg_quantized_model():
    in_channel = 3
    input_shape = [1, in_channel, 6, 6]
    out_channel = 4
    output_shape = [1, out_channel, 6, 6]
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.int8, input_shape)},
            "output": {"y": (np.int8, output_shape)},
            "initializer": {
                "w": (np.int8, [out_channel, in_channel, 1, 1]),
                "a": (np.int8, [1, out_channel, 6, 6]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("QLinearConv", ["x", "s", "z", "w", "s", "z", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("DequantizeLinear", ["a", "s", "z"], ["2"]),
                ("Add", ["1", "2"], ["3"]),
                ("QuantizeLinear", ["3", "s", "z"], ["y"]),
            ],
        },
        # gives check=False to bypass
        # onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: \
        # [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation \
        # for the node QLinearConv_0:QLinearConv(10) error.
        check=False,
    )


def _make_sandwiced_model():
    input_shape = [1, 8]
    output_shape = input_shape
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.int8, input_shape)},
            "output": {"y": (np.int8, output_shape)},
            "initializer": {
                "a": (np.int8, [input_shape[1]]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("DequantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["a", "s", "z"], ["1"]),
                ("Add", ["0", "1"], ["2"]),
                ("QuantizeLinear", ["2", "s", "z"], ["y"]),
            ],
        }
    )


def _make_partially_sandwiced_model():
    input_shape = [1, 8]
    output_shape = input_shape
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.int8, input_shape)},
            "output": {"y": (np.int8, output_shape)},
            "initializer": {
                "a": (np.float32, [input_shape[1]]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("DequantizeLinear", ["x", "s", "z"], ["0"]),
                ("Add", ["0", "a"], ["1"]),
                ("QuantizeLinear", ["1", "s", "z"], ["y"]),
            ],
        }
    )


def _make_partially_sandwiched_top_model():
    input_shape = [1, 8]
    output_shape = input_shape
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.int8, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.int8, [input_shape[1]]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("DequantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["a", "s", "z"], ["1"]),
                ("Add", ["0", "1"], ["y"]),
            ],
        }
    )


def _make_partially_sandwiced_bot_model():
    input_shape = [1, 8]
    output_shape = input_shape
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.int8, output_shape)},
            "initializer": {
                "a": (np.float32, [input_shape[1]]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("Add", ["x", "a"], ["0"]),
                ("QuantizeLinear", ["0", "s", "z"], ["y"]),
            ],
        }
    )


def _make_sandwiced_i64_model():
    input_shape = [1, 8]
    output_shape = input_shape
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.int64, input_shape)},
            "output": {"y": (np.int64, output_shape)},
            "initializer": {
                "a": (np.int64, input_shape),
            },
            "node": [
                ("Add", ["x", "a"], ["y"]),
            ],
        }
    )


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
