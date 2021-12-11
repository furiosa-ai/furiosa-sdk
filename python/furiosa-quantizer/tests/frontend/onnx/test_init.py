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
        model = onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        graph = shape_inference.infer_shapes(model).graph
        test_node = graph.node[2]
        self.assertTrue(_is_sandwiched(test_node, *parse_onnx_graph(graph)))

    def test__is_sandwiched2(self):
        model = onnx.load(Path(__file__).resolve().parent / "partially_sandwiched(top).onnx")
        graph = shape_inference.infer_shapes(model).graph
        test_node = graph.node[2]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(graph)))

    def test__is_sandwiched3(self):
        model = onnx.load(Path(__file__).resolve().parent / "partially_sandwiched.onnx")
        graph = shape_inference.infer_shapes(model).graph
        test_node = graph.node[1]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(graph)))

    def test__is_sandwiched4(self):
        model = onnx.load(Path(__file__).resolve().parent / "partially_sandwiched(bot).onnx")
        graph = shape_inference.infer_shapes(model).graph
        test_node = graph.node[0]
        self.assertFalse(_is_sandwiched(test_node, *parse_onnx_graph(graph)))

    def test__is_fully_quantized_in_dfg_mode(self):
        model = onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        graph = shape_inference.infer_shapes(model).graph
        self.assertTrue(_is_fully_quantized_in_dfg_mode(graph, *parse_onnx_graph(graph)))

    def test__is_fully_quantized_in_dfg_mode2(self):
        model = onnx.load(Path(__file__).resolve().parent / "quantized_in_dfg_mode.onnx")
        graph = shape_inference.infer_shapes(model).graph
        self.assertTrue(_is_fully_quantized_in_dfg_mode(graph, *parse_onnx_graph(graph)))

    def test__is_fully_quantized_in_dfg_mode3(self):
        model = onnx.load(Path(__file__).resolve().parent / "partially_sandwiched(top).onnx")
        graph = shape_inference.infer_shapes(model).graph
        self.assertFalse(_is_fully_quantized_in_dfg_mode(graph, *parse_onnx_graph(graph)))

    def test__is_fully_quantized_in_fake_quant_mode(self):
        model = onnx.load(Path(__file__).resolve().parent / "sandwiched.onnx")
        graph = shape_inference.infer_shapes(model).graph
        self.assertTrue(_is_fully_quantized_in_fake_quant_mode(graph, *parse_onnx_graph(graph)))

    def test__is_fully_quantized_in_fake_quant_mode2(self):
        model = onnx.load(Path(__file__).resolve().parent / "partially_sandwiched(bot).onnx")
        graph = shape_inference.infer_shapes(model).graph
        self.assertFalse(_is_fully_quantized_in_fake_quant_mode(graph, *parse_onnx_graph(graph)))

    def test__is_fully_quantized(self):
        model = onnx.load(Path(__file__).resolve().parent / "quantized_in_dfg_mode.onnx")
        self.assertTrue(_is_fully_quantized(model))


if __name__ == "__main__":
    unittest.main()
