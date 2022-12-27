import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import Pattern_1
from furiosa.quantizer.frontend.onnx.transformer.utils import get_attribute
from tests.frontend.onnx.transformer import TestTransformer


class TestConvertConv1dToConv2d(TestTransformer):
    def test_case1(self):
        in_channel = 8
        input_shape = [1, in_channel, 5]
        out_channel = 4
        output_shape = [1, out_channel, 7]
        kernel_size = 4
        group = 2
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel // group, kernel_size]),
                "b": (np.float32, [out_channel]),
            },
            "node": [
                (
                    "Conv",
                    ["x", "w", "b"],
                    ["y"],
                    {
                        "pads": [2, 3],
                        "kernel_shape": [kernel_size],
                        "group": group,
                        "dilations": [1],
                        "strides": [1],
                    },
                ),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Reshape', 'Conv', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
        conv = next(node for node in trans_model.graph.node if node.name == 'Conv_1')
        self.check_attribute(get_attribute(conv.attribute, attr_name="pads"), [0, 2, 0, 3])
        self.check_attribute(
            get_attribute(conv.attribute, attr_name="kernel_shape"), [1, kernel_size]
        )
        self.check_attribute(get_attribute(conv.attribute, attr_name="group"), group)
        self.check_attribute(get_attribute(conv.attribute, attr_name="dilations"), [1, 1])
        self.check_attribute(get_attribute(conv.attribute, attr_name="strides"), [1, 1])

    def test_case2(self):
        in_channel = 3
        input_shape = [1, in_channel, 3]
        out_channel = 5
        output_shape = [1, out_channel, 1]
        kernel_size = 1
        group = 1
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel // group, kernel_size]),
            },
            "node": [
                (
                    "Conv",
                    ["x", "w"],
                    ["y"],
                    {
                        "dilations": [2],
                        "strides": [3],
                    },
                ),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Reshape', 'Conv', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
        conv = next(node for node in trans_model.graph.node if node.name == 'Conv_1')
        self.check_attribute(get_attribute(conv.attribute, attr_name="dilations"), [1, 2])
        self.check_attribute(get_attribute(conv.attribute, attr_name="strides"), [1, 3])
