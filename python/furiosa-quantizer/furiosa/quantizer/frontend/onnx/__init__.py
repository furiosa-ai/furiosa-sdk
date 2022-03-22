from collections import defaultdict
import itertools
from typing import IO, Callable, Dict, List, Optional, Text, Tuple

import numpy as np
import onnx

__DOMAIN__ = ''
__OPSET_VERSION__ = 12

from furiosa.quantizer.frontend.onnx import calibrate
from furiosa.quantizer.frontend.onnx.quantizer import quantizer
from furiosa.quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import (
    ConvertConv1dToConv2d,
)
from furiosa.quantizer.frontend.onnx.transformer.eliminate_redundant_shape_pattern import (
    EliminateRedundantShapePattern,
)
from furiosa.quantizer.frontend.onnx.transformer.fuse_batchnorm import FuseBatchNorm
from furiosa.quantizer.frontend.onnx.transformer.fuse_conv import FuseConv
from furiosa.quantizer.frontend.onnx.transformer.fuse_depth_to_space import FuseDepthToSpace
from furiosa.quantizer.frontend.onnx.transformer.fuse_gather_matmul import FuseGatherMatMul
from furiosa.quantizer.frontend.onnx.transformer.fuse_gelu import FuseGELU
from furiosa.quantizer.frontend.onnx.transformer.fuse_layer_normalization import (
    FuseLayerNormalization,
)
from furiosa.quantizer.frontend.onnx.transformer.fuse_lp_normalization import FuseLpNormalization
from furiosa.quantizer.frontend.onnx.transformer.fuse_pad import FusePad
from furiosa.quantizer.frontend.onnx.transformer.fuse_redundant_reshape_pattern import (
    FuseRedundantReshapePattern,
)
from furiosa.quantizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa.quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa.quantizer.frontend.onnx.utils.version_checker import CheckVersion

_CONV = "Conv"
_DEQUANTIZE_LINEAR = "DequantizeLinear"
_MAT_MUL = "MatMul"
_QUANTIZE_LINEAR = "QuantizeLinear"
_Q_LINEAR_CONV = "QLinearConv"
_Q_LINEAR_MAT_MUL = "QLinearMatMul"


def _transform(
    transformers: List[Callable[[onnx.ModelProto], onnx.ModelProto]], model: onnx.ModelProto
) -> onnx.ModelProto:
    for transform in transformers:
        model = transform(model)
    return model


def _inference_shape(model: onnx.ModelProto) -> onnx.ModelProto:
    return InferenceShape(model).inference_shape()


def _reify(model: onnx.ModelProto) -> onnx.ModelProto:
    transformers = [
        ConvertConv1dToConv2d().transform,
        FuseConv().transform,
        FusePad().transform,
        FuseBatchNorm().transform,
        FuseDepthToSpace().transform,
        FuseGELU().transform,
        FuseLayerNormalization().transform,
        FuseLpNormalization().transform,
        FuseRedundantReshapePattern().transform,
        FuseGatherMatMul().transform,
        EliminateRedundantShapePattern().transform,
    ]
    return _transform(transformers, model)


def optimize_model(
    model: onnx.ModelProto, input_shapes: Optional[Dict[str, List[int]]] = None
) -> onnx.ModelProto:
    model = _transform([CheckVersion().transform], model)
    model = _transform([PolishModel(input_shapes).transform], model)

    # TODO check if graph_transform should apply.
    model = _transform([_reify], model)
    return model


def quantize(
    model: onnx.ModelProto,
    per_channel: bool,
    static: bool,
    mode: quantizer.QuantizationMode,
    dynamic_ranges: Dict[str, Tuple[float, float]],
) -> onnx.ModelProto:
    return quantizer.FuriosaONNXQuantizer(
        model, per_channel, static, mode, dynamic_ranges
    ).quantize()


def post_training_quantize(
    model: onnx.ModelProto,
    dataset: List[Dict[str, np.ndarray]],
    per_channel: bool = True,
    check_idempotency: bool = False,
) -> onnx.ModelProto:
    """Post-training-quantizes an ONNX model with a calibration dataset.

    Args:
        model: An ONNX model to quantize.
        dataset: A calibration dataset.
        per_channel: If per_channel is True, Conv's filters are
          per-channel quantized. Otherwise, they are per-tensor
          quantized.
    Returns:
        An ONNX model post-training-quantized with the calibration
        dataset.
    """
    if check_idempotency:
        if _is_fully_quantized(model):
            return model
        if any(node.op_type in [_DEQUANTIZE_LINEAR, _QUANTIZE_LINEAR] for node in model.graph.node):
            raise ValueError(
                "an ONNX model with DequantizeLinear or QuantizeLinear is not supported yet."
            )

    model = optimize_model(model)
    ranges = calibrate.calibrate(model, dataset)
    return quantize(model, per_channel, True, quantizer.QuantizationMode.DFG, ranges)


def post_training_quantization_with_random_calibration(
    model: onnx.ModelProto,
    per_channel: bool,
    static: bool,
    mode: quantizer.QuantizationMode,
    num_data: int = 8,
    check_idempotency: bool = False,
) -> onnx.ModelProto:
    if not static:
        raise Exception("Currently only supports static quantization.")

    if check_idempotency:
        if _is_fully_quantized(model):
            return model
        if any(node.op_type in [_DEQUANTIZE_LINEAR, _QUANTIZE_LINEAR] for node in model.graph.node):
            raise ValueError(
                "an ONNX model with DequantizeLinear or QuantizeLinear is not supported yet."
            )

    model = optimize_model(model)
    dynamic_ranges = calibrate.calibrate_with_random_data(model, num_data)
    return quantize(model, per_channel, static, mode, dynamic_ranges)


def parse_onnx_graph(
    model: onnx.ModelProto,
) -> Tuple[
    Dict[str, onnx.ValueInfoProto], Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]
]:
    model = onnx.shape_inference.infer_shapes(model)

    value_infos = {
        value_info.name: value_info
        for value_info in itertools.chain(
            model.graph.input, model.graph.output, model.graph.value_info
        )
    }
    # append graph.initializer's value_info to value_infos
    value_infos.update(
        (init.name, onnx.helper.make_tensor_value_info(init.name, init.data_type, init.dims))
        for init in model.graph.initializer
    )
    assert all(
        tensor in value_infos
        for node in model.graph.node
        for tensor in itertools.chain(node.input, node.output)
    ), "ONNX Shape inference did not work well on some tensor(s). All value_infos must be inferenced to proceed."
    producer = {tensor: node for node in model.graph.node for tensor in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for tensor in node.input:
            consumers[tensor].append(node)

    return value_infos, producer, consumers


def _is_fully_quantized(model: onnx.ModelProto) -> bool:
    """Return `True` if the ONNX `graph` is already quantized."""
    return _is_fully_quantized_in_dfg_mode(
        model.graph, *parse_onnx_graph(model)
    ) or _is_fully_quantized_in_fake_quant_mode(model.graph, *parse_onnx_graph(model))


def _is_fully_quantized_in_dfg_mode(
    graph: onnx.GraphProto,
    value_infos: Dict[str, onnx.ValueInfoProto],
    producer: Dict[str, onnx.NodeProto],
    consumers: Dict[str, List[onnx.NodeProto]],
) -> bool:
    return all(
        node.op_type in [_DEQUANTIZE_LINEAR, _QUANTIZE_LINEAR, _Q_LINEAR_CONV, _Q_LINEAR_MAT_MUL]
        or (
            node.op_type not in [_CONV, _MAT_MUL]
            and _is_sandwiched(node, value_infos, producer, consumers)
        )
        for node in graph.node
    )


def _is_fully_quantized_in_fake_quant_mode(
    graph: onnx.GraphProto,
    value_infos: Dict[str, onnx.ValueInfoProto],
    producer: Dict[str, onnx.NodeProto],
    consumers: Dict[str, List[onnx.NodeProto]],
) -> bool:
    return all(
        node.op_type in [_DEQUANTIZE_LINEAR, _QUANTIZE_LINEAR]
        or _is_sandwiched(node, value_infos, producer, consumers)
        for node in graph.node
    )


def _is_sandwiched(
    node: onnx.NodeProto,
    value_infos: Dict[str, onnx.ValueInfoProto],
    producer: Dict[str, onnx.NodeProto],
    consumers: Dict[str, List[onnx.NodeProto]],
) -> bool:
    return all(
        value_infos[tensor].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or (tensor in producer and producer[tensor].op_type == _DEQUANTIZE_LINEAR)
        for tensor in node.input
    ) and all(
        value_infos[tensor].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or (
            tensor in consumers
            and all(consumer.op_type == _QUANTIZE_LINEAR for consumer in consumers[tensor])
        )
        for tensor in node.output
    )
