from typing import Dict, List, Optional, Text, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort

from furiosa.optimizer.frontend.onnx import __DOMAIN__, __OPSET_VERSION__


def make_onnx_model_from_model_desc(
    model_desc: Dict,
    doc_string: str = "",
    producer_name: str = "",
    check: bool = True,
    infer_shape: bool = True,
) -> onnx.ModelProto:  # pylint: disable=no-member
    """
    type hints for model_desc

    model_desc = {
        "input": Dict[str, Tuple[np.dtype, List[int]]],
        "output": Dict[str, Tuple[np.dtype, List[int]]],
        "initializer": Dict[str, Union[Tuple[np.dtype, List[int]], np.ndarray]],
        "node": List[Tuple[str, List[str], List[str], Dict]],
        "opsetid": List[Tuple[str, str]],
    }
    """
    input_desc = model_desc["input"]
    output_desc = model_desc["output"]
    init_desc = model_desc.get("initializer", {})
    node_desc = model_desc.get("node", [])
    opsetid_desc = model_desc.get("opsetid", [(__DOMAIN__, __OPSET_VERSION__)])

    inputs = _make_value_info_list(input_desc)
    outputs = _make_value_info_list(output_desc)
    inits = _make_initializer_list(init_desc)
    nodes = _make_node_list(
        [desc if isinstance(desc[-1], dict) else (*desc, {}) for desc in node_desc]
    )
    opsetids = _make_opsetid_list(opsetid_desc)
    graph = _make_graph(nodes, inputs, outputs, inits, value_info=None, doc_string=doc_string)
    model = _make_model(graph, opsetids, check, producer_name)

    if infer_shape:
        model = onnx.shape_inference.infer_shapes(model)

    return model


def _make_value_info_list(
    vi_desc: Dict[str, Tuple[np.dtype, List[Union[Text, int]]]]
) -> List[onnx.ValueInfoProto]:  # pylint: disable=no-member
    return [
        onnx.helper.make_tensor_value_info(
            name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(np_dtype)], shape
        )
        for name, (np_dtype, shape) in vi_desc.items()
    ]


def _make_initializer_list(
    init_desc: Dict[str, Union[Tuple[np.dtype, List[Union[Text, int]]], np.array]]
) -> List[onnx.TensorProto]:  # pylint: disable=no-member
    initailizers = []
    for name, arg in init_desc.items():
        if isinstance(arg, tuple):
            dtype, shape = arg
            initailizers.append(onnx.numpy_helper.from_array(_random_generator(dtype, shape), name))
        elif isinstance(arg, np.ndarray):
            np_arr = arg
            initailizers.append(onnx.numpy_helper.from_array(np_arr, name))
        else:
            raise TypeError(repr(type(arg)))

    return initailizers


def _make_node_list(
    node_desc: List[Tuple[str, List[str], List[str], Dict]]
) -> List[onnx.NodeProto]:  # pylint: disable=no-member
    return [
        onnx.helper.make_node(op_type, inputs, outputs, name=f'{op_type}_{idx}', **attrs)
        for idx, (op_type, inputs, outputs, attrs) in enumerate(node_desc)
    ]


def _make_opsetid_list(
    opsetids: List[Tuple[str, str]]
) -> List[onnx.OperatorSetIdProto]:  # pylint:disable=no-member
    return [onnx.helper.make_opsetid(domain, version) for (domain, version) in opsetids]


def _make_graph(
    nodes: List[onnx.NodeProto],  # pylint: disable=no-member
    inputs: List[onnx.ValueInfoProto],  # pylint: disable=no-member
    outputs: List[onnx.ValueInfoProto],  # pylint: disable=no-member
    initializer: Optional[List[onnx.TensorProto]] = None,  # pylint: disable=no-member
    value_info: Optional[List[onnx.ValueInfoProto]] = None,  # pylint: disable=no-member
    name: str = "graph",
    doc_string: str = "",
) -> onnx.GraphProto:  # pylint: disable=no-member
    graph_def = onnx.helper.make_graph(
        nodes, name, inputs, outputs, initializer, doc_string, value_info
    )
    return graph_def


def _make_model(
    graph: onnx.GraphProto,  # pylint: disable=no-member
    opset_imports: List[onnx.OperatorSetIdProto],  # pylint: disable=no-member
    check: bool = True,
    producer_name: str = "",
) -> onnx.ModelProto:  # pylint: disable=no-member
    model = onnx.helper.make_model(graph, opset_imports=opset_imports, producer_name=producer_name)
    if check:
        onnx.checker.check_model(model)
        ort.InferenceSession(model.SerializeToString())
    return model


def _random_generator(np_dtype, shape):
    rng = np.random.default_rng()
    if np.issubdtype(np_dtype, np.floating):
        return rng.standard_normal(shape, dtype=np.dtype(np_dtype))
    if np.issubdtype(np_dtype, np.integer):
        numeric_info = np.iinfo(np_dtype)
        return rng.integers(
            low=numeric_info.min, high=numeric_info.max, size=shape, dtype=np.dtype(np_dtype)
        )
    raise TypeError(repr(np_dtype))
