import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_conv import Pattern_1, Pattern_2, Pattern_3
from tests.frontend.onnx.transformer import TestTransformer


# TODO 1. Generate test model that does not meet conditions for conv fusion
class TestFuseConv(TestTransformer):
    def test_case1(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [8, 6]),
                "a": (np.float32, [1, 6]),
            },
            "node": [
                ("MatMul", ["x", "w"], ["0"]),
                ("Add", ["0", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [8, 6]),
            },
            "node": [
                ("Gemm", ["x", "w"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [8, 6]), "b": (np.float32, [6])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shape = [4, 8]
        output_shape = [6, 8]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [6, 4]), "b": (np.float32, [8])},
            "node": [
                ("Gemm", ["w", "x", "b"], ["y"]),
            ],
        }

        _, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Gemm'])

    def test_case5(self):
        input_shape = [8, 4]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [8, 6]), "b": (np.float32, [6])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"], {"alpha": 1.1, "transA": 1}),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Transpose', 'Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case6(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [6, 8]), "b": (np.float32, [6])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"], {"beta": 2.3, "transB": 1}),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case7(self):
        input_shape = [4, 8]
        output_shape = [8, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [6, 4]), "b": (np.float32, [6])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"], {"transA": 1, "transB": 1}),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Transpose', 'Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case8(self):
        in_channel = 8
        input_shape = [2, in_channel, 1, 1]
        out_channel = 6
        output_shape = [2, out_channel, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "a": (np.float32, [1, out_channel, 1, 1]),
            },
            "node": [("Conv", ["x", "w"], ["0"]), ("Add", ["0", "a"], ["y"])],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case9(self):
        in_channel = 8
        input_shape = [2, in_channel, 1, 1]
        out_channel = 6
        output_shape = [2, out_channel, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
                "a": (np.float32, [1, out_channel, 1, 1]),
            },
            "node": [("Conv", ["x", "w", "b"], ["0"]), ("Add", ["a", "0"], ["y"])],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case10(self):
        in_channel = 16
        input_shape = [2, in_channel, 8, 8]
        out_channel = 4
        output_shape = [2, out_channel, 8, 8]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
            },
            "node": [("Conv", ["x", "w", "b"], ["0"]), ("Add", ["0", "0"], ["y"])],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)

        self.check_graph_node(trans_model, op_types=['Conv', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case11(self):
        in_channel = 8
        input_shape = [2, in_channel, 1, 1]
        out_channel = 6
        output_shape = [2, out_channel, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
                "a": (np.float32, [1]),
            },
            "node": [("Conv", ["x", "w", "b"], ["0"]), ("Add", ["a", "0"], ["y"])],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case12(self):
        in_channel = 8
        input_shape = [2, in_channel, 1, 1]
        out_channel = 6
        output_shape = [2, out_channel, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
                "a": (np.float32, [out_channel, 1, 1]),
            },
            "node": [("Conv", ["x", "w", "b"], ["0"]), ("Add", ["a", "0"], ["y"])],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case13(self):
        in_channel = 8
        input_shape = [2, in_channel, 1, 1]
        out_channel = 6
        output_shape = [2, out_channel, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
                "a": (np.float32, [2, out_channel, 1, 1]),
            },
            "node": [("Conv", ["x", "w", "b"], ["0"]), ("Add", ["a", "0"], ["y"])],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case14(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [8, 6]),
                "a": (np.float32, [1]),
            },
            "node": [
                ("MatMul", ["x", "w"], ["0"]),
                ("Add", ["0", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case15(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [8, 6]),
                "a": (np.float32, [4, 6]),
            },
            "node": [
                ("MatMul", ["x", "w"], ["0"]),
                ("Add", ["0", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['MatMul', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case16(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [8, 6]), "b": (np.float32, [1])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Unsqueeze', 'Conv', 'Squeeze'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case17(self):
        input_shape = [4, 8]
        output_shape = [4, 6]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"w": (np.float32, [8, 6]), "b": (np.float32, [4, 6])},
            "node": [
                ("Gemm", ["x", "w", "b"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Gemm'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
