import numpy as np

from furiosa.quantizer.frontend.onnx.quantizer.eliminate_clipper import (
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
    Pattern_5,
    Pattern_6,
)
from tests.frontend.onnx.transformer import TestTransformer


class TestEliminateClipper(TestTransformer):
    def test_case1(self):
        in_channel = 16
        input_shape = [2, in_channel, 4, 4]
        out_channel = 8
        output_shape = [2, out_channel, 4, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("QuantizeLinear", ["w", "s", "z"], ["w_quant"]),
                ("DequantizeLinear", ["w_quant", "s", "z"], ["w_dequant"]),
                ("Conv", ["1", "w_dequant"], ["2"]),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("Relu", ["4"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)

    def test_case2(self):
        in_channel = 6
        input_shape = [3, in_channel, 10, 10]
        out_channel = 9
        output_shape = [3, out_channel, 8, 8]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "s": (np.float32, []),
                "z": (np.int8, []),
                "min": np.array(0.0, dtype=np.float32),
                "max": np.array(6.0, dtype=np.float32),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("QuantizeLinear", ["w", "s", "z"], ["w_quant"]),
                ("DequantizeLinear", ["w_quant", "s", "z"], ["w_dequant"]),
                ("Conv", ["1", "w_dequant"], ["2"]),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("QuantizeLinear", ["min", "s", "z"], ["min_quant"]),
                ("DequantizeLinear", ["min_quant", "s", "z"], ["min_dequant"]),
                ("QuantizeLinear", ["max", "s", "z"], ["max_quant"]),
                ("DequantizeLinear", ["max_quant", "s", "z"], ["max_dequant"]),
                ("Clip", ["4", "min_dequant", "max_dequant"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [1, 3, 8, 8]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.float32, input_shape),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("QuantizeLinear", ["a", "s", "z"], ["a_quant"]),
                ("DequantizeLinear", ["a_quant", "s", "z"], ["a_dequant"]),
                ("Add", ["1", "a_dequant"], ["2"]),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("Relu", ["4"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shape = [1, 3, 8, 8]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.float32, input_shape),
                "s": (np.float32, []),
                "z": (np.int8, []),
                "min": np.array(-1.0, dtype=np.float32),
                "max": np.array(1.0, dtype=np.float32),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("QuantizeLinear", ["a", "s", "z"], ["a_quant"]),
                ("DequantizeLinear", ["a_quant", "s", "z"], ["a_dequant"]),
                ("Add", ["1", "a_dequant"], ["2"]),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("QuantizeLinear", ["min", "s", "z"], ["min_quant"]),
                ("DequantizeLinear", ["min_quant", "s", "z"], ["min_dequant"]),
                ("QuantizeLinear", ["max", "s", "z"], ["max_quant"]),
                ("DequantizeLinear", ["max_quant", "s", "z"], ["max_dequant"]),
                ("Clip", ["4", "min_dequant", "max_dequant"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_4)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)

    def test_case5(self):
        in_channel = out_channel = 32
        input_shape = output_shape = [1, in_channel]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "s": (np.float32, []),
                "z": (np.int8, []),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("Unsqueeze", ["1"], ["2"], {"axes": [2, 3]}),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("QuantizeLinear", ["w", "s", "z"], ["w_quant"]),
                ("DequantizeLinear", ["w_quant", "s", "z"], ["w_dequant"]),
                ("Conv", ["4", "w_dequant"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["7"]),
                ("Squeeze", ["7"], ["8"], {"axes": [2, 3]}),
                ("QuantizeLinear", ["8", "s", "z"], ["9"]),
                ("DequantizeLinear", ["9", "s", "z"], ["10"]),
                ("Relu", ["10"], ["11"]),
                ("QuantizeLinear", ["11", "s", "z"], ["12"]),
                ("DequantizeLinear", ["12", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_5)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Unsqueeze',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Squeeze',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)

    def test_case6(self):
        """
        This tests Pattern_6
        """
        in_channel = out_channel = 32
        input_shape = output_shape = [1, in_channel]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "s": (np.float32, []),
                "z": (np.int8, []),
                "min": np.array(0.0, dtype=np.float32),
                "max": np.array(1.0, dtype=np.float32),
            },
            "node": [
                ("QuantizeLinear", ["x", "s", "z"], ["0"]),
                ("DequantizeLinear", ["0", "s", "z"], ["1"]),
                ("Unsqueeze", ["1"], ["2"], {"axes": [2, 3]}),
                ("QuantizeLinear", ["2", "s", "z"], ["3"]),
                ("DequantizeLinear", ["3", "s", "z"], ["4"]),
                ("QuantizeLinear", ["w", "s", "z"], ["w_quant"]),
                ("DequantizeLinear", ["w_quant", "s", "z"], ["w_dequant"]),
                ("Conv", ["4", "w_dequant"], ["5"]),
                ("QuantizeLinear", ["5", "s", "z"], ["6"]),
                ("DequantizeLinear", ["6", "s", "z"], ["7"]),
                ("Squeeze", ["7"], ["8"], {"axes": [2, 3]}),
                ("QuantizeLinear", ["8", "s", "z"], ["9"]),
                ("DequantizeLinear", ["9", "s", "z"], ["10"]),
                ("QuantizeLinear", ["min", "s", "z"], ["min_quant"]),
                ("DequantizeLinear", ["min_quant", "s", "z"], ["min_dequant"]),
                ("QuantizeLinear", ["max", "s", "z"], ["max_quant"]),
                ("DequantizeLinear", ["max_quant", "s", "z"], ["max_dequant"]),
                ("Clip", ["10", "min_dequant", "max_dequant"], ["11"]),
                ("QuantizeLinear", ["11", "s", "z"], ["12"]),
                ("DequantizeLinear", ["12", "s", "z"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_6)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Unsqueeze',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Squeeze',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.check_value_info(trans_model)
