import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import Pattern_1
from tests.frontend.onnx.transformer import TestTransformer


class TestConvertConv1dToConv2d(TestTransformer):
    def test_case1(self):
        input_shape = [1, 16]
        output_shape = [1, 14, 1, 1]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "shape": np.array([1, 1, -1], dtype=np.int64),
                "w": (np.float32, [1, 1, 3]),
                "shape1": np.array([1, -1, 1, 1], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "shape"], ["1"]),
                ("Conv", ["1", "w"], ["2"]),
                ("Reshape", ["2", "shape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Reshape', 'Conv', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
