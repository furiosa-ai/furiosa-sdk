import itertools
import logging
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import onnx
from onnx.helper import make_model, make_opsetid, make_tensor_value_info

from furiosa.quantizer.frontend.onnx import __DOMAIN__, __OPSET_VERSION__
from furiosa.quantizer.frontend.onnx.quantizer.utils import __PRODUCER__

logger = logging.getLogger('Furiosa-Quantizer')
logging.basicConfig(level=logging.INFO)


def name_nodes(model):
    for idx, node in enumerate(model.graph.node):
        node.name = f'{node.op_type}_{idx}'

    return model


def eliminate_unused_initializer(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    This function eliminates every initializers not used by node input,
    regardless of any graph fields they are defined in.
    """
    node_input_names = set(tensor_name for node in model.graph.node for tensor_name in node.input)
    used_initializers = [
        tensor for tensor in model.graph.initializer if tensor.name in node_input_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(used_initializers)
    return model


def eliminate_unused_input(model):
    node_input_names = set(tensor_name for node in model.graph.node for tensor_name in node.input)
    graph_output_names = set(value_info.name for value_info in model.graph.output)
    used_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name in node_input_names or value_info.name in graph_output_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(used_inputs)
    return model


def eliminate_unused_output(model):
    node_output_names = set(tensor_name for node in model.graph.node for tensor_name in node.output)
    graph_input_names = set(value_info.name for value_info in model.graph.input)
    used_outputs = [
        value_info
        for value_info in model.graph.output
        if value_info.name in node_output_names or value_info.name in graph_input_names
    ]
    del model.graph.output[:]
    model.graph.output.extend(used_outputs)
    return model


def eliminate_unused_value_info(model):
    node_output_names = set(tensor_name for node in model.graph.node for tensor_name in node.output)
    graph_output_names = set(value_info.name for value_info in model.graph.output)
    used_value_infos = [
        value_info
        for value_info in model.graph.value_info
        if value_info.name in node_output_names and value_info.name not in graph_output_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(used_value_infos)
    return model


def eliminate_unused_protos(model):
    funcs = [
        eliminate_unused_initializer,
        eliminate_unused_input,
        eliminate_unused_output,
        eliminate_unused_value_info,
    ]

    for func in funcs:
        model = func(model)

    return model


def eliminate_initializer_from_graph_input(model: onnx.ModelProto) -> onnx.ModelProto:
    initializer = set(init.name for init in model.graph.initializer)
    graph_input = [
        value_info for value_info in model.graph.input if value_info.name not in initializer
    ]
    del model.graph.input[:]
    model.graph.input.extend(graph_input)

    return model


def include_initializer_to_graph_input(model: onnx.ModelProto) -> onnx.ModelProto:
    input_value_names = [inp.name for inp in model.graph.input]
    for init in model.graph.initializer:
        if init.name not in input_value_names:
            value_info = make_tensor_value_info(init.name, init.data_type, init.dims)
            model.graph.input.append(value_info)
            # do not append duplicated initializer to graph input
            input_value_names.append(init.name)
    return model


def rebuild_model(
    model: onnx.ModelProto,
    new_nodes: List[onnx.NodeProto],
    eliminate: bool = True,
    renaming: bool = True,
):
    # remove all nodes and re-make model.graph based on newly given nodes.
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    default_opset = make_opsetid(__DOMAIN__, __OPSET_VERSION__)
    model = make_model(model.graph, opset_imports=[default_opset])

    # eliminate all unused protos such as initializer, input, output, and value_info.
    if eliminate:
        model = eliminate_initializer_from_graph_input(model)
        model = eliminate_unused_protos(model)

    # rename node.name
    if renaming:
        model = name_nodes(model)
    model.producer_name = __PRODUCER__
    return model


def fix_batch_size_as_one(model):
    """
    fix batch_size = 1 if dim_param is given.
    """
    for value_info in model.graph.input:
        try:
            batch_dim = value_info.type.tensor_type.shape.dim[0]
        except IndexError:
            continue

        if batch_dim.dim_param:
            logger.info(
                "Dynamic batch size is detected at input_name: %s. "
                "Fix batch_size=1 for valid shape inference.",
                value_info.name,
            )
            value_info.type.tensor_type.shape.dim[0].dim_value = 1

    return model


def make_initializer_name_unique(model):
    # Renames Operators' initializers, if necessary, to make their names unique
    initializer = {init.name: init for init in model.graph.initializer}
    del model.graph.initializer[:]
    for node in model.graph.node:
        for idx, node_input in enumerate(node.input):
            if node_input not in initializer:
                continue

            tensor = onnx.TensorProto()
            tensor.CopyFrom(initializer[node_input])
            # HACK: This attempts to give the initializer a new unique name.
            # Although it is unlikely, there is a possibility that the new
            # name is already occupied by a tensor in the model.
            tensor.name = f"{node_input}_{node.output[0]}_{idx}"
            node.input[idx] = tensor.name
            model.graph.initializer.append(tensor)

    return model


T = TypeVar('T')


def fixed_point(x: T, functions: Iterable[Callable[[T], T]]) -> T:
    while True:
        init = x
        for func in functions:
            x = func(x)
        if x == init:
            return x


def get_attribute(
    attrs: Iterable[onnx.AttributeProto], attr_name: str, default: Optional[Any] = None
) -> Any:
    return next(
        (onnx.helper.get_attribute_value(attr) for attr in attrs if attr.name == attr_name), default
    )


def make_unhashables_unique(values):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)

    return seen


def is_op_type(op_type: str, target_op_types: Iterable[str]) -> bool:
    return op_type in target_op_types


def check_value_info(model: onnx.ModelProto) -> None:
    initializer = {init.name: init for init in model.graph.initializer}
    value_info = {
        vi.name: vi
        for vi in itertools.chain(model.graph.value_info, model.graph.input, model.graph.output)
    }
    tensor_names = set(
        tensor_name for node in model.graph.node for tensor_name in (*node.input, *node.output)
    )
    for name in tensor_names:
        if (
            name in initializer or not name
        ):  # empty name indicates that optional input is unspecified
            continue

        if name not in value_info:
            raise ValueError(
                f'value_info of {name} is missing. Optimize model before quantization.'
            )
        try:
            if not value_info[name].type.tensor_type.HasField('elem_type'):
                raise ValueError(
                    f'elem_type of {name} in value_info is missing. Optimize model before quantization, or shape inference failed.'
                )
            if value_info[name].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                logger.warning(
                    'elem_type of %s(%s) is not FLOAT: Model might be already quantized.',
                    name,
                    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                        value_info[name].type.tensor_type.elem_type
                    ],
                )
            if not value_info[name].type.tensor_type.HasField(
                'shape'
            ):  # when shape inference failed, shape field does not exist, unlike empty shape.dim array for scalar.
                raise ValueError(
                    f'shape of {name} in value_info is missing. Optimize model before quantization, or shape inference failed.'
                )
        except AttributeError as e:
            raise AttributeError(
                f'{e} (ValueInfoProto is incomplete. Optimize model before quantization, or shape inference failed.)'
            ) from None
