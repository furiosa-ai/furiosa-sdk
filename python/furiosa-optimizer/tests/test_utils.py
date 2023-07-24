import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.utils import (
    eliminate_unused_input,
    eliminate_unused_output,
)


def test_eliminate_unused_input(make_model):
    input_shape = [4, 5]
    model_desc = {
        "input": {"x": (np.float32, input_shape)},
        "output": {"x": (np.float32, input_shape)},
    }

    model = make_model(model_desc)
    model = eliminate_unused_input(model)

    # Graph input should not be removed if graph input = graph output
    assert model.graph.input


def test_eliminate_unused_output(make_model):
    input_shape = [1, 2, 3]
    model_desc = {
        "input": {"x": (np.float32, input_shape)},
        "output": {"x": (np.float32, input_shape)},
    }

    model = make_model(model_desc)
    model = eliminate_unused_output(model)

    # Graph output should not be removed if graph input = graph output
    assert model.graph.output
