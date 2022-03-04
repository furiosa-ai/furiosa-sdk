import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_gelu import FuseGELU
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseGELU(TestTransformer):
    def test_case1(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only GELU operator
        """
        input_shape = [1, 1, 4, 4]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "reciprocal": np.array(1.4142135381698608, dtype=np.float32),
                "a": np.array(1, dtype=np.float32),
                "w": np.array(0.5, dtype=np.float32),
            },
            "node": [
                ("Div", ["x", "reciprocal"], ["0"]),
                ("Erf", ["0"], ["1"]),
                ("Add", ["1", "a"], ["2"]),
                ("Mul", ["x", "2"], ["3"]),
                ("Mul", ["3", "w"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, FuseGELU())
        self.check_graph_node(trans_model, op_types=["Gelu"])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        """
        Test whether the original model is well transformed for multi operator model,
         which contains operators other than Gelu
        """
        input_shape = [2, 4, 8, 8, 16]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "reciprocal": np.array(1.4142135381698608, dtype=np.float32),
                "a": np.array(1, dtype=np.float32),
                "w": np.array(0.5, dtype=np.float32),
                "s": (np.float32, input_shape),
            },
            "node": [
                ("Sub", ["x", "s"], ["0"]),
                ("Div", ["0", "reciprocal"], ["1"]),
                ("Erf", ["1"], ["2"]),
                ("Add", ["2", "a"], ["3"]),
                ("Mul", ["x", "3"], ["4"]),
                ("Mul", ["4", "w"], ["5"]),
                ("Sub", ["5", "s"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, FuseGELU())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
