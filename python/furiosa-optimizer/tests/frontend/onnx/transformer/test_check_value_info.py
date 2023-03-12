import unittest

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx.transformer import utils
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model


class TestCheckValueInfo(unittest.TestCase):
    def test_shape_inference_no_dims(self):
        input_shape = [1, 4, 1, 1]
        output_shape = [4]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {},
            "node": [
                ("Squeeze", ["x"], ["x_1"]),
                ("Unsqueeze", ["x_1"], ["x_2"], {"axes": [0, 2, 3]}),
                ("Squeeze", ["x_2"], ["x_3"]),
                ("Unsqueeze", ["x_3"], ["x_4"], {"axes": [0, 2, 3]}),
                ("Squeeze", ["x_4"], ["y"]),
            ],
            "opsetid": [("", 12)],
        }
        model = make_onnx_model(model_desc)
        self.assertRaisesRegex(ValueError, r"shape of(\w)*", utils.check_value_info, model)

    def test_no_shape_inference(self):
        model_desc = {
            "input": {"x": (np.float32, [2, 2])},
            "output": {"z": (np.float32, [2, 2])},
            "node": [("Add", ["x", "x"], ["y"]), ("Add", ["x", "y"], ["z"])],
        }
        model = make_onnx_model(model_desc, infer_shape=False)
        self.assertRaisesRegex(ValueError, r"value_info of(\w)*", utils.check_value_info, model)

    def test_no_elem_type(self):
        def _make_y_value_info():
            value_info_y = onnx.ValueInfoProto()  # pylint: disable=no-member
            value_info_y.name = "y"
            type_proto_y = onnx.TypeProto()  # pylint: disable=no-member
            type_proto_tensor_y = onnx.TypeProto.Tensor()  # pylint: disable=no-member
            type_proto_y.tensor_type.CopyFrom(type_proto_tensor_y)
            tensor_shape_proto_y = onnx.TensorShapeProto()  # pylint: disable=no-member
            dim = onnx.TensorShapeProto.Dimension()  # pylint: disable=no-member
            dim.dim_value = 2
            tensor_shape_proto_y.dim.extend([dim, dim])
            type_proto_y.tensor_type.shape.CopyFrom(tensor_shape_proto_y)
            value_info_y.type.CopyFrom(type_proto_y)
            return value_info_y

        model_desc = {
            "input": {"x": (np.float32, [2, 2])},
            "output": {"z": (np.float32, [2, 2])},
            "node": [("Add", ["x", "x"], ["y"]), ("Add", ["x", "y"], ["z"])],
        }
        model = make_onnx_model(model_desc, infer_shape=False)
        model.graph.value_info.append(_make_y_value_info())
        self.assertRaisesRegex(ValueError, r"elem_type of(\w)*", utils.check_value_info, model)
