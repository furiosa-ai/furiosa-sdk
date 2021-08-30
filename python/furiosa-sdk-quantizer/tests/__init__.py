from typing import List, Tuple, Dict, Optional

import io
import onnx
from onnx import helper
# To prevent seg fault in Mac OS X
import onnxruntime
import torch
import numpy as np

from furiosa.quantizer.frontend.onnx import __DOMAIN__, __OPSET_VERSION__


def torch_to_onnx(torch_model: torch.nn.Module, input_shapes: List[Tuple[int, ...]]) -> onnx.ModelProto:
    torch_model.eval()
    f = io.BytesIO()
    dummies = []
    for shape in input_shapes:
        dummies.append(torch.ones(shape, dtype=torch.float32))

    torch.onnx.export(torch_model, *dummies, f, opset_version=__OPSET_VERSION__)
    return onnx.load_model(io.BytesIO(f.getvalue()), helper.ModelProto)


def make_test_model(input_shapes: List[Tuple[int, ...]],
                    output_shapes: List[Tuple[int, ...]],
                    attributes: Dict,
                    init_shapes: Optional[List[Tuple[int, ...]]] = None,
                    op_type: Optional[str] = 'TestOp',
                    check_model: Optional[bool] = False) \
        -> Tuple[onnx.ModelProto, onnx.NodeProto]:
    inputs, input_names = make_test_value_info(input_shapes, name='X')
    outputs, output_names = make_test_value_info(output_shapes, name='Y')

    inits = None
    if init_shapes:
        inits, init_names = make_test_init(init_shapes, name='Z')
        input_names += init_names

    # Create the model (NodeProto)
    node_def = helper.make_node(
        op_type=op_type,
        inputs=input_names,
        outputs=output_names,
        **attributes,
        name='%s_0' % op_type
    )

    kwargs = {
        'nodes': [node_def],
        'name': 'test-model',
        'inputs': inputs,
        'outputs': outputs,
        'initializer': inits,
    }

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        **kwargs
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid(__DOMAIN__, __OPSET_VERSION__)])

    if check_model:
        onnx.checker.check_model(model_def)
        from onnx import shape_inference
        model_def = shape_inference.infer_shapes(model_def)

    return model_def, node_def


def make_test_model_with_init_val(input_shapes: List[Tuple[int, ...]],
                                  output_shapes: List[Tuple[int, ...]],
                                  attributes: Dict,
                                  init_values: List[np.ndarray],
                                  op_type: Optional[str] = 'TestOp',
                                  check_model: Optional[bool] = False) \
        -> Tuple[onnx.ModelProto, onnx.NodeProto]:
    inputs, input_names = make_test_value_info(input_shapes, name='X')
    outputs, output_names = make_test_value_info(output_shapes, name='Y')

    inits, init_names = make_test_init_with_val(init_values, name='Z')
    input_names += init_names

    # Create the model (NodeProto)
    node_def = helper.make_node(
        op_type=op_type,
        inputs=input_names,
        outputs=output_names,
        **attributes,
        name='%s_0' % op_type
    )

    kwargs = {
        'nodes': [node_def],
        'name': 'test-model',
        'inputs': inputs,
        'outputs': outputs,
        'initializer': inits,
    }

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        **kwargs
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid(__DOMAIN__, __OPSET_VERSION__)])

    if check_model:
        onnx.checker.check_model(model_def)
        from onnx import shape_inference
        model_def = shape_inference.infer_shapes(model_def)

    return model_def, node_def


def make_test_value_info(shapes: List[Tuple[int, ...]], name: str) \
        -> Tuple[List[onnx.ValueInfoProto], List[str]]:
    vi_protos = list()
    vi_names = list()
    for idx, shape in enumerate(shapes):
        vi = helper.make_tensor_value_info(name='%s_%d' % (name, idx),
                                           elem_type=helper.TensorProto.FLOAT,
                                           shape=shape)
        vi_protos.append(vi)
        vi_names.append(vi.name)

    return vi_protos, vi_names


def make_test_init(shapes: List[Tuple[int, ...]], name: str) \
        -> Tuple[List[onnx.TensorProto], List[str]]:
    init_protos = list()
    init_names = list()

    for idx, shape in enumerate(shapes):
        val = np.ones(shape, dtype=np.float32)
        init = helper.make_tensor(name='%s_%d' % (name, idx),
                                  data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype],
                                  dims=val.shape,
                                  vals=val.flatten())
        init_protos.append(init)
        init_names.append(init.name)

    return init_protos, init_names


def make_test_init_with_val(values: List[np.ndarray], name: str) \
        -> Tuple[List[onnx.TensorProto], List[str]]:
    init_protos = list()
    init_names = list()

    for idx, val in enumerate(values):
        init = helper.make_tensor(name='%s_%d' % (name, idx),
                                  data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype],
                                  dims=val.shape,
                                  vals=val.flatten())
        init_protos.append(init)
        init_names.append(init.name)

    return init_protos, init_names
