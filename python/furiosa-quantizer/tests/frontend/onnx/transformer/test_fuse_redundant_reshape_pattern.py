import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_redundant_reshape_pattern import (
    FuseRedundantReshapePattern,
    Pattern_1,
    Pattern_2,
    Pattern_3,
)
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseRedundantReshapePattern(TestTransformer):
    def test_case1(self):
        input_shape = [16, 8]
        output_shape = [8, 2, 8]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "shape": np.array([4, 4, 8], dtype=np.int64),
                "shape1": np.array([8, 2, 8], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "shape"], ["0"]),
                ("Reshape", ["0", "shape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [16, 8]
        output_shape = [32, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "shape": np.array([4, 4, 8], dtype=np.int64),
                "shape1": np.array([8, 2, 8], dtype=np.int64),
                "shape2": np.array([32, 4], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "shape"], ["0"]),
                ("Reshape", ["0", "shape1"], ["1"]),
                ("Reshape", ["1", "shape2"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [16, 8]
        output_shape = [4, 4, 2, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "shape": np.array([8, 2, 8], dtype=np.int64),
                "shape1": np.array([32, 4], dtype=np.int64),
                "shape2": np.array([4, 4, 2, 4], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "shape"], ["0"]),
                ("Reshape", ["0", "shape1"], ["1"]),
                ("Reshape", ["1", "shape2"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, FuseRedundantReshapePattern())
        self.check_graph_node(trans_model, op_types=['Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shape = [16, 1, 8]
        output_shape = [1, 16, 8]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "node": [
                ("Squeeze", ["x"], ["0"], {"axes": [1]}),
                ("Unsqueeze", ["0"], ["y"], {"axes": [0]}),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
