import numpy as np
import pytest

from furiosa.optimizer.frontend.onnx.utils.version_checker import CheckVersion


def test_case(make_model):
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
    orig_model = make_model(model_desc)
    # convert opset 11 to 12
    trans_model = CheckVersion(opset_version=new_opset).transform(orig_model)

    assert trans_model.opset_import[0].version == new_opset


def test_case1(make_model):
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
    orig_model = make_model(model_desc)
    # convert opset 12 to 13
    trans_model = CheckVersion(opset_version=new_opset).transform(orig_model)

    assert trans_model.opset_import[0].version == new_opset


def test_case2():
    new_opset = 11

    # if target opset < 12, CheckVersion should raise ValueError
    with pytest.raises(ValueError):
        CheckVersion(new_opset)


def test_case3():
    new_opset = 14

    # if target opset > 13, CheckVersion should raise VlueError
    with pytest.raises(ValueError):
        CheckVersion(new_opset)
