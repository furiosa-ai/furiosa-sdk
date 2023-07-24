from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx.transformer.infer_squeeze_axes import InferSqueezeAxes
from furiosa.optimizer.frontend.onnx.transformer.utils import check_value_info, fixed_point
from furiosa.optimizer.frontend.onnx.utils.inference_shape import InferenceShape


def test_case1(make_model, check_output_value):
    input_shape = [1, 4, 1, 1]
    output_shape = [8]
    opsetid = ("", 12)
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
        "opsetid": [opsetid],
    }

    orig_model, trans_model = _make_test_model(make_model, model_desc, {"x": input_shape})

    check_output_value(orig_model, trans_model, [input_shape])
    check_value_info(trans_model)
    assert [0, 2, 3] == trans_model.graph.node[0].attribute[0].ints


def test_case2(make_model, check_output_value):
    input_shape = [1, 4, 1, 1]
    output_shape = [4]
    opsetid = ("", 12)
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
        "opsetid": [opsetid],
    }

    orig_model, trans_model = _make_test_model(make_model, model_desc, {"x": input_shape})

    check_output_value(orig_model, trans_model, [input_shape])
    check_value_info(trans_model)
    assert [0, 2, 3] == trans_model.graph.node[0].attribute[0].ints
    assert [0, 2, 3] == trans_model.graph.node[2].attribute[0].ints
    assert [0, 2, 3] == trans_model.graph.node[4].attribute[0].ints


def _make_test_model(
    make_model, model_desc: Dict, input_shapes: Mapping[str, List[Optional[int]]]
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:  # pylint: disable=no-member
    orig_model = make_model(model_desc)
    trans_model = fixed_point(
        orig_model,
        [
            lambda model: InferenceShape(model).inference_shape(input_shapes),
            InferSqueezeAxes().transform,
        ],
    )

    return orig_model, trans_model
