import unittest

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx.utils.version_checker import CheckVersion
from tests.frontend.onnx.transformer import make_onnx_model


class TestCheckVersion(unittest.TestCase):
    def test_case(self):
        input_shape = [8]
        output_shape = [8]
        opsetid = ("", 11)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "node": [("Relu", ["x"], ["y"])],
            "opsetid": [opsetid],
        }
        new_opset = 12
        orig_model = make_onnx_model(model_desc)
        # convert opset 11 to 12
        trans_model = CheckVersion(opset_version=new_opset).transform(orig_model)
        self.check_opset_version(trans_model, new_opset)

    def test_case1(self):
        input_shape = [8]
        output_shape = [8]
        opsetid = ("", 12)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "node": [("Relu", ["x"], ["y"])],
            "opsetid": [opsetid],
        }
        new_opset = 13
        orig_model = make_onnx_model(model_desc)
        # convert opset 12 to 13
        trans_model = CheckVersion(opset_version=new_opset).transform(orig_model)
        self.check_opset_version(trans_model, new_opset)

    def test_case2(self):
        new_opset = 11
        # if target opset < 12, CheckVersion should raise ValueError
        self.assertRaises(ValueError, CheckVersion, new_opset)

    def test_case3(self):
        new_opset = 14
        # if target opset > 13, CheckVersion should raise VlueError
        self.assertRaises(ValueError, CheckVersion, new_opset)

    def check_opset_version(self, model: onnx.ModelProto, opset: int):  # pylint: disable=no-member
        self.assertEqual(model.opset_import[0].version, opset)
