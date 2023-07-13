import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.convert_prelu_to_relu import Pattern_1, Pattern_2
from furiosa.optimizer.frontend.onnx.transformer.utils import check_value_info


def test_case1(make_transformer_model, check_output_value, check_graph_node):
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

    orig_model, trans_model = make_transformer_model(model_desc, Pattern_1)

    check_output_value(orig_model, trans_model, [input_shape])
    check_graph_node(trans_model, op_types=["Relu", "Mul", "Mul", "Add"])
    check_value_info(trans_model)


def test_case2(make_transformer_model, check_output_value, check_graph_node):
    input_shape = [1, 3, 4, 4]
    slope_shape = [1, 3, 1, 1]
    output_shape = [1, 3, 4, 4]
    opsetid = ("", 13)
    model_desc = {
        "input": {
            "data": (np.float32, input_shape),
            "half_slope": (np.float32, slope_shape),
        },
        "output": {"output": (np.float32, output_shape)},
        "node": [
            ("Add", ["half_slope", "half_slope"], ["slope"]),
            ("PRelu", ["data", "slope"], ["output"]),
        ],
        "opsetid": [opsetid],
    }

    orig_model, trans_model = make_transformer_model(model_desc, Pattern_2)

    check_output_value(orig_model, trans_model, [input_shape, slope_shape])
    check_graph_node(trans_model, op_types=["Add", "Relu", "Sub", "Mul", "Mul", "Add"])
    check_value_info(trans_model)
