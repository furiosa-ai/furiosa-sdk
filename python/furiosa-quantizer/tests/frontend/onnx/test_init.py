#!/usr/bin/env python3
from pathlib import Path
import pickle
import unittest

import onnx
from onnx import shape_inference

from furiosa.quantizer.frontend.onnx import (
    _is_fully_quantized,
    _is_fully_quantized_in_dfg_mode,
    _is_fully_quantized_in_fake_quant_mode,
    _is_sandwiched,
    parse_onnx_graph,
    post_training_quantize,
)


class ONNXTest(unittest.TestCase):
    def test_post_training_quantize(self):
        model = onnx.load(Path(__file__).resolve().parent / "efficientdet_d0-f3276ba8.onnx")
        # val2017-10.pickle contains a calibration dataset that consists
        # of 10 images in the COCO validation dataset [val2017.zip][].
        #
        # [val2017.zip]: http://images.cocodataset.org/zips/val2017.zip
        with open(Path(__file__).resolve().parent / "val2017-10.pickle", "rb") as f:
            dataset = pickle.load(f)
        model = post_training_quantize(model, dataset)

    def test__is_sandwiched(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        )
        test_node = model.graph.node[2]
        self.assertTrue(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched2(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "sandwiched_i64.onnx")
        )
        test_node = model.graph.node[0]
        self.assertTrue(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched3(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "partially_sandwiched_top.onnx")
        )
        test_node = model.graph.node[2]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched4(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "partially_sandwiched.onnx")
        )
        test_node = model.graph.node[1]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_sandwiched5(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "partially_sandwiched_bot.onnx")
        )
        test_node = model.graph.node[0]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        )
        self.assertTrue(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode2(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "quantized_in_dfg_mode.onnx")
        )
        self.assertTrue(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_dfg_mode3(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "partially_sandwiched_top.onnx")
        )
        self.assertFalse(_is_fully_quantized_in_dfg_mode(model.graph, *parse_onnx_graph(model)))

    def test__is_fully_quantized_in_fake_quant_mode(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        )
        self.assertTrue(
            _is_fully_quantized_in_fake_quant_mode(model.graph, *parse_onnx_graph(model))
        )

    def test__is_fully_quantized_in_fake_quant_mode2(self):
        model = shape_inference.infer_shapes(
            onnx.load(Path(__file__).resolve().parent / "partially_sandwiched_bot.onnx")
        )
        self.assertFalse(
            _is_fully_quantized_in_fake_quant_mode(model.graph, *parse_onnx_graph(model))
        )

    def test__is_fully_quantized(self):
        model = onnx.load(Path(__file__).resolve().parent / "quantized_in_dfg_mode.onnx")
        self.assertTrue(_is_fully_quantized(model))


if __name__ == "__main__":
    unittest.main()
