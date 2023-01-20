import unittest

import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.utils import (
    eliminate_unused_input,
    eliminate_unused_output,
)
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model


class TestUtils(unittest.TestCase):
    def test_eliminate_unused_input(self):
        input_shape = [4, 5]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"x": (np.float32, input_shape)},
        }

        model = make_onnx_model(model_desc)
        model = eliminate_unused_input(model)
        # graph input should not be removed if graph input = graph output
        self.assertTrue(model.graph.input)

    def test_eliminate_unused_output(self):
        input_shape = [1, 2, 3]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"x": (np.float32, input_shape)},
        }

        model = make_onnx_model(model_desc)
        model = eliminate_unused_output(model)
        # graph output should not be removed if graph input = graph output
        self.assertTrue(model.graph.output)
