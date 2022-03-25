import logging
from typing import Callable, Iterable, TypeVar

import onnx
from onnx import numpy_helper
from onnx.helper import make_model, make_opsetid, make_tensor_value_info

from furiosa.quantizer.frontend.onnx import __DOMAIN__, __OPSET_VERSION__
from furiosa.quantizer.frontend.onnx.quantizer.utils import __PRODUCER__

logger = logging.getLogger('Furiosa-Quantizer')
logging.basicConfig(level=logging.INFO)


def name_nodes(model):
    for idx, node in enumerate(model.graph.node):
        node.name = f'{node.op_type}_{idx}'

    return model


def eliminate_unused_initializer(model):
    node_input_names = set(tensor_name for node in model.graph.node for tensor_name in node.input)
    unused_initializers = (
        tensor for tensor in model.graph.initializer if tensor.name not in node_input_names
    )
    for unused_initializer in unused_initializers:
        model.graph.initializer.remove(unused_initializer)
    return model


def eliminate_unused_input(model):
    node_input_names = set(tensor_name for node in model.graph.node for tensor_name in node.input)
    unused_inputs = (
        value_info for value_info in model.graph.input if value_info.name not in node_input_names
    )
    for unused_input in unused_inputs:
        model.graph.input.remove(unused_input)
    return model


def eliminate_unused_output(model):
    node_output_names = set(tensor_name for node in model.graph.node for tensor_name in node.output)
    unused_outputs = (
        value_info for value_info in model.graph.output if value_info.name not in node_output_names
    )
    for unused_output in unused_outputs:
        model.graph.output.remove(unused_output)
    return model


def eliminate_unused_value_info(model):
    node_output_names = set(tensor_name for node in model.graph.node for tensor_name in node.output)
    graph_output_names = set(value_info.name for value_info in model.graph.output)
    unused_value_infos = (
        value_info
        for value_info in model.graph.value_info
        if value_info.name not in node_output_names or value_info.name in graph_output_names
    )
    for unused_value_info in unused_value_infos:
        model.graph.value_info.remove(unused_value_info)
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


def include_initializer_to_graph_input(model):
    input_value_names = [inp.name for inp in model.graph.input]
    for init in model.graph.initializer:
        if init.name not in input_value_names:
            dims = numpy_helper.to_array(init).shape
            value_info = make_tensor_value_info(init.name, init.data_type, dims)
            model.graph.input.append(value_info)

            # do not append duplicated initializer to graph input
            input_value_names.append(init.name)

    return model


def rebuild_model(model, new_nodes, eliminate=True, renaming=True):
    # remove all nodes and re-make model.graph based on newly given nodes.
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    default_opset = make_opsetid(__DOMAIN__, __OPSET_VERSION__)
    model = make_model(model.graph, opset_imports=[default_opset])

    # eliminate all unused protos such as initializer, input, output, and value_info.
    if eliminate:
        model = eliminate_unused_protos(model)

    # include initializer to graph input
    model = include_initializer_to_graph_input(model)

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
    model.graph.ClearField('initializer')
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
