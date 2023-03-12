#!/usr/bin/env python3
from typing import Optional, Sequence
import unittest

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx import optimize_model
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model
from tests.frontend.onnx.transformer import TestTransformer


class ONNXTest(TestTransformer):
    def test_opset_version(self):
        model = _make_test_model(opset_version=11)
        new_opset = 12
        model = optimize_model(model, opset_version=new_opset)
        self.check_opset_version(model, new_opset)

    def test_opset_version_1(self):
        model = _make_test_model(opset_version=12)
        new_opset = 13
        model = optimize_model(model, opset_version=new_opset)
        self.check_opset_version(model, new_opset)

    def check_opset_version(self, model: onnx.ModelProto, opset: int):  # pylint: disable=no-member
        self.assertEqual(model.opset_import[0].version, opset)


def _make_test_model(
    opset_version: int,
    input_shape: Optional[Sequence[int]] = None,
    output_shape: Optional[Sequence[int]] = None,
) -> onnx.ModelProto:  # pylint: disable=no-member
    if input_shape is None:
        input_shape = [2, 8]
    if output_shape is None:
        output_shape = [2, 6]
    in_channel = input_shape[1]
    out_channel = output_shape[1]
    opsetid = ("", opset_version)
    return make_onnx_model(
        model_desc={
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [in_channel, out_channel]),
                "a": (np.float32, [1, out_channel]),
            },
            "node": [
                ("MatMul", ["x", "w"], ["0"]),
                ("Add", ["0", "a"], ["y"]),
            ],
            "opsetid": [opsetid],
        }
    )


if __name__ == "__main__":
    unittest.main()
