import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.fuse_batchnorm import (
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
)
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseBatchNorm(TestTransformer):
    def test_case1(self):
        in_channel = 3
        input_shape = [1, in_channel, 4, 4]
        out_channel = 4
        output_shape = [1, out_channel, 3, 3]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("BatchNormalization", ["0", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        in_channel = 3
        input_shape = [1, in_channel, 4, 4]
        out_channel = 4
        output_shape = [1, out_channel, 3, 3]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Relu", ["0"], ["1"]),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_4)
        self.check_graph_node(trans_model, op_types=['Conv', 'Relu', 'Mul', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3(self):
        in_channel = 4
        input_shape = [1, in_channel, 4, 4]
        out_channel = 8
        output_shape = [1, out_channel, 2, 2]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w1": (np.float32, [1, in_channel]),
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
                "a": (np.float32, output_shape),
            },
            "node": [
                ("Mul", ["x", "w1"], ["0"]),
                ("Conv", ["0", "w", "b"], ["1"]),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["2"]),
                ("Add", ["2", "a"], ["y"]),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Mul', 'Conv', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case4(self):
        in_channel = 5
        input_shape = [1, in_channel, 4, 4]
        out_channel = 2
        output_shape = [1, out_channel, 2, 2]

        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w1": (np.float32, input_shape),
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
                "a": (np.float32, output_shape),
            },
            "node": [
                ("Mul", ["x", "w1"], ["0"]),
                ("Conv", ["0", "w", "b"], ["1"]),
                ("Relu", ["1"], ["2"]),
                ("BatchNormalization", ["2", "beta", "gamma", "mu", "var"], ["3"]),
                ("Add", ["a", "3"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_4)
        self.check_graph_node(trans_model, op_types=['Mul', 'Conv', 'Relu', 'Mul', 'Add', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case5(self):
        """
        This tests Pattern_3
        """
        in_channel = 3
        input_shape = [1, in_channel, 8, 8]
        out_channel = 4
        output_shape = [1, out_channel, 6, 6]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "b": (np.float32, [out_channel]),
                "w1": (np.float32, [1, out_channel, 1, 1]),
                "a": (np.float32, [1, out_channel, 1, 1]),
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Mul", ["0", "w1"], ["1"]),
                ("Add", ["1", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case5_1(self):
        """
        This tests Pattern_3
        """
        in_channel = 3
        input_shape = [1, in_channel, 8, 8]
        out_channel = 4
        output_shape = [1, out_channel, 6, 6]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "b": (np.float32, [out_channel]),
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Mul", ["0", "0"], ["1"]),
                ("Add", ["1", "1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv', 'Mul', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case5_2(self):
        in_channel = 4
        input_shape = [1, in_channel, 4, 4]
        out_channel = 4
        output_shape = [1, out_channel, 2, 2]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "w1": (np.float32, [1, out_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Mul", ["0", "w1"], ["1"]),
                ("Add", ["1", "1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Conv', 'Mul', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case6(self):
        in_channel = 3
        input_shape = [1, in_channel, 4, 4]
        out_channel = 4
        output_shape = [1, out_channel, 5, 5]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [in_channel, out_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                ("ConvTranspose", ["x", "w", "b"], ["0"]),
                ("BatchNormalization", ["0", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['ConvTranspose'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case7(self):
        in_channel = 3
        input_shape = [1, in_channel, 4, 4]
        out_channel = 4
        output_shape = [1, out_channel, 5, 5]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [in_channel, out_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                ("ConvTranspose", ["x", "w", "b"], ["0"]),
                ("Relu", ["0"], ["1"]),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['ConvTranspose', 'Relu', 'BatchNormalization'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case8(self):
        in_channel = 4
        input_shape = [1, in_channel, 4, 4]
        out_channel = 8
        output_shape = [1, out_channel, 5, 5]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w1": (np.float32, input_shape),
                "w": (np.float32, [in_channel, out_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
                "a": (np.float32, output_shape),
            },
            "node": [
                ("Mul", ["x", "w1"], ["0"]),
                ("ConvTranspose", ["0", "w", "b"], ["1"]),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["2"]),
                ("Add", ["2", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['Mul', 'ConvTranspose', 'Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case9(self):
        in_channel = 5
        input_shape = [1, in_channel, 4, 4]
        out_channel = 2
        output_shape = [1, out_channel, 5, 5]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w1": (np.float32, input_shape),
                "w": (np.float32, [in_channel, out_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
                "a": (np.float32, output_shape),
            },
            "node": [
                ("Mul", ["x", "w1"], ["1"]),
                ("ConvTranspose", ["1", "w", "b"], ["2"]),
                ("Relu", ["2"], ["3"]),
                ("BatchNormalization", ["3", "beta", "gamma", "mu", "var"], ["4"]),
                ("Add", ["4", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(
            trans_model, op_types=['Mul', 'ConvTranspose', 'Relu', 'BatchNormalization', 'Add']
        )
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case10(self):
        in_channel = 16
        input_shape = [1, in_channel, 6, 6]
        out_channel = 2
        output_shape = [1, out_channel, 3, 3]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 4, 4]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                (
                    "Conv",
                    ["x", "w"],
                    ["1"],
                    {
                        "dilations": [1, 1],
                        "group": 1,
                        "kernel_shape": [4, 4],
                        "pads": [1, 1, 1, 1],
                        "strides": [2, 2],
                    },
                ),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case11(self):
        in_channel = 16
        input_shape = [1, in_channel, 4, 4]
        out_channel = 2
        output_shape = [1, out_channel, 8, 8]
        scale, B, input_mean, input_var = _bn_param_generator(num_features=out_channel)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [in_channel, out_channel, 4, 4]),
                "beta": scale,
                "gamma": B,
                "mu": input_mean,
                "var": input_var,
            },
            "node": [
                (
                    "ConvTranspose",
                    ["x", "w"],
                    ["1"],
                    {
                        "dilations": [1, 1],
                        "group": 1,
                        "kernel_shape": [4, 4],
                        "pads": [1, 1, 1, 1],
                        "strides": [2, 2],
                    },
                ),
                ("BatchNormalization", ["1", "beta", "gamma", "mu", "var"], ["y"]),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=['ConvTranspose'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)


def _bn_param_generator(num_features):
    """
    returns scale, B, input_mean and input_var
    as defined in https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-12
    """
    rng = np.random.default_rng()
    return (
        rng.standard_normal(num_features, dtype=np.float32),
        rng.standard_normal(num_features, dtype=np.float32),
        rng.standard_normal(num_features, dtype=np.float32),
        rng.standard_normal(num_features, dtype=np.float32) ** 2,
    )
