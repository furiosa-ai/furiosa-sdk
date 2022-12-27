import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.convert_negative_pads_to_slice import (
    ConvertNegativePadsToSlice,
)
from tests.frontend.onnx.transformer import TestTransformer


class TestConvertNegativePadsToSlice(TestTransformer):
    def test_case1(self):
        input_shape = [1, 2, 4, 4]
        output_shape = [1, 2, 4, 4]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "pads": np.array([0, 0, -1, -1, 0, 0, 1, 1], dtype=np.int64),
                "constant_value": np.array([0], dtype=np.float32),
            },
            "node": [
                ("Pad", ["data", "pads", "constant_value"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, ConvertNegativePadsToSlice())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_graph_node(trans_model, op_types=['Slice', 'Pad'])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [1, 2, 4, 4]
        output_shape = [1, 2, 2, 1]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "pads": np.array([0, 0, -1, -1, 0, 0, -1, -2], dtype=np.int64),
                "constant_value": np.array([0], dtype=np.float32),
            },
            "node": [
                ("Pad", ["data", "pads", "constant_value"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, ConvertNegativePadsToSlice())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_graph_node(trans_model, op_types=['Slice'])
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [1, 2, 4, 4]
        output_shape = [1, 2, 0, 4]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "pads": np.array([0, 0, -2, -1, 0, 0, -2, 1], dtype=np.int64),
                "constant_value": np.array([0], dtype=np.float32),
            },
            "node": [
                ("Pad", ["data", "pads", "constant_value"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, ConvertNegativePadsToSlice())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_graph_node(trans_model, op_types=['Slice', 'Pad'])
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shape = [1, 2, 4, 4]
        output_shape = [1, 2, 1, 5]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "pads": np.array([0, 0, -1, -2, 0, 0, -2, 3], dtype=np.int64),
                "constant_value": np.array([0], dtype=np.float32),
            },
            "node": [
                ("Pad", ["data", "pads", "constant_value"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        orig_model, trans_model = self.make_test_model(model_desc, ConvertNegativePadsToSlice())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_graph_node(trans_model, op_types=['Slice', 'Pad'])
        self.check_value_info(trans_model)

    def test_case5(self):
        input_shape = [1, 2, 4, 4]
        output_shape = [1, 2, 4, 4]
        opsetid = ("", 13)
        model_desc = {
            "input": {"data": (np.float32, input_shape)},
            "output": {"output": (np.float32, output_shape)},
            "initializer": {
                "pads": np.array([0, 0, -4, -1, 0, 0, 4, 1], dtype=np.int64),
                "constant_value": np.array([0], dtype=np.float32),
            },
            "node": [
                ("Pad", ["data", "pads", "constant_value"], ["output"]),
            ],
            "opsetid": [opsetid],
        }

        _orig_model, trans_model = self.make_test_model(model_desc, ConvertNegativePadsToSlice())
        # Don't check output value, but check that transformation is not happened.
        # Though orig_model and trans_model are same, they output 'nan' intermittently,
        # which leads to failure of check_output_value.
        self.check_graph_node(trans_model, op_types=['Pad'])
        self.check_value_info(trans_model)
