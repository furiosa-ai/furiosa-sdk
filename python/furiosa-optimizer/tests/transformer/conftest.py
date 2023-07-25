from typing import Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from furiosa.optimizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.optimizer.interfaces.transformer import Transformer


@pytest.fixture
def make_transformer_model(make_model):
    def inner(
        model_desc: Dict,
        transformer: Union[Transformer, Type[ONNXTransformer]],
    ):
        orig_model = make_model(model_desc)
        copy_model = onnx.ModelProto()  # pylint: disable=no-member
        copy_model.CopyFrom(orig_model)

        if isinstance(transformer, Transformer):
            trans_model = transformer.transform(copy_model)
        elif issubclass(transformer, ONNXTransformer):
            trans_model = transformer(copy_model).transform()
        else:
            raise TypeError(repr(transformer))

        return orig_model, trans_model

    return inner


@pytest.fixture
def check_graph_node():
    def inner(model: onnx.ModelProto, op_types: List[str]):  # pylint: disable=no-member
        assert len(model.graph.node) == len(op_types)
        for node, op_type in zip(model.graph.node, op_types):
            assert node.op_type == op_type

    return inner


@pytest.fixture
def check_output_value():
    def inner(
        orig_model: onnx.ModelProto,  # pylint: disable=no-member
        trans_model: onnx.ModelProto,  # pylint: disable=no-member
        input_shapes: Iterable[Sequence[int]],
        data: Optional[List[np.ndarray]] = None,
    ):
        if data is None:
            rng = np.random.default_rng()
            data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        actual = _run_onnx_model_flatten_output(orig_model, data)
        expected = _run_onnx_model_flatten_output(trans_model, data)

        for act, exp in zip(actual, expected):
            assert len(act) == len(exp)
            for a, b in zip(act, exp):
                assert a == pytest.approx(b, 2), f"{data}"

    return inner


# def check_attribute(actual, expected):
#     assert actual == expected


def _run_onnx_model(
    model: onnx.ModelProto, input_arrays: List[np.ndarray]  # pylint: disable=no-member
) -> List[np.ndarray]:
    sess = ort.InferenceSession(model.SerializeToString())
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    feed_dict = dict(zip(input_names, input_arrays))
    outputs = sess.run(output_names, input_feed=feed_dict)

    return outputs


def _run_onnx_model_flatten_output(
    model: onnx.ModelProto, input_arrays: List[np.ndarray]  # pylint: disable=no-member
) -> List[List[float]]:
    outputs = _run_onnx_model(model, input_arrays)
    return [val.flatten().tolist() for val in outputs]
