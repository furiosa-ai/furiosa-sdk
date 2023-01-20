import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.convert_prelu_to_relu import Pattern_1, Pattern_2
from tests.frontend.onnx.transformer import TestTransformer


class TestConvertPReluToRelu(TestTransformer):
    def test_case1(self):
        input_shape = [1, 3, 4, 4]
        output_shape = [1, 3, 4, 4]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "slope": np.array([[[1]], [[-2]], [[3]]], dtype=np.float32),
            },
            "node": [
                ("PRelu", ["data", "slope"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_graph_node(trans_model, op_types=['Relu', 'Mul', 'Mul', 'Add'])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [1, 3, 4, 4]
        slope_shape = [1, 3, 1, 1]
        output_shape = [1, 3, 4, 4]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape), "half_slope": (np.float32, slope_shape)},
            "output": {"output": (np.float32, output_shape)},
            "node": [
                ("Add", ["half_slope", "half_slope"], ["slope"]),
                ("PRelu", ["data", "slope"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_output_value(orig_model, trans_model, [input_shape, slope_shape])
        self.check_graph_node(trans_model, op_types=['Add', 'Relu', 'Sub', 'Mul', 'Mul', 'Add'])
        self.check_value_info(trans_model)
