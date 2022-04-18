from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.transformer.infer_squeeze_axes import InferSqueezeAxes
from furiosa.quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model
from tests.frontend.onnx.transformer import TestTransformer


class TestInferSqueezeAxes(TestTransformer):
    def test_case1(self):
        input_shape = [1, 4, 1, 1]
        output_shape = [8]
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [4, 8]),
                "b": (np.float32, [8]),
            },
            "node": [
                ("Squeeze", ["x"], ["x_1"]),
                ("MatMul", ["x_1", "w"], ["z"]),
                ("Add", ["z", "b"], ["y"]),
            ],
        }

        orig_model, trans_model = _make_test_model(model_desc, {"x": input_shape})
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_attribute([0, 2, 3], trans_model.graph.node[0].attribute[0].ints)
        self.check_value_info(trans_model)

    def test_case2(self):
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
        }

        orig_model, trans_model = _make_test_model(model_desc, {"x": input_shape})
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_attribute([0, 2, 3], trans_model.graph.node[0].attribute[0].ints)
        self.check_attribute([0, 2, 3], trans_model.graph.node[2].attribute[0].ints)
        self.check_attribute([0, 2, 3], trans_model.graph.node[4].attribute[0].ints)
        self.check_value_info(trans_model)


def _make_test_model(
    model_desc: Dict, input_shapes: List[Sequence[int]]
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
    orig_model = make_onnx_model(model_desc)
    trans_model = utils.fixed_point(
        orig_model,
        [
            lambda model: InferenceShape(model).inference_shape(input_shapes),
            InferSqueezeAxes().transform,
        ],
    )

    return orig_model, trans_model
