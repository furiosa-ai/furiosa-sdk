from collections import defaultdict
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
from furiosa.quantizer.frontend.onnx.utils.version_checker import CheckVersion

_DEQUANTIZE_LINEAR = "DequantizeLinear"
_QUANTIZE_LINEAR = "QuantizeLinear"
_Q_LINEAR_CONV = "QLinearConv"
_Q_LINEAR_MAT_MUL = "QLinearMatMul"


class AlreadyQuantizedError(ValueError):
    """
    Exception raised if given model is partially quantized.
    """

    def __init__(self, op_type: str) -> None:
        assert op_type in [
            _DEQUANTIZE_LINEAR,
            _QUANTIZE_LINEAR,
            _Q_LINEAR_CONV,
            _Q_LINEAR_MAT_MUL,
        ], repr(op_type)
        super().__init__(
            "furiosa-quantizer cannot proceed with a quantized model. "
            "The model seems to be at least partially quantized "
            f"since it includes {op_type}."
        )


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
    dataset: Iterable[Dict[str, np.ndarray]],
    per_channel: bool = True,
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

    _verify_not_quantized(model)
    model = optimize_model(model)
    ranges = calibrate.calibrate(model, dataset)
    return quantize(model, per_channel, True, quantizer.QuantizationMode.DFG, ranges)


def post_training_quantization_with_random_calibration(
    model: onnx.ModelProto,
    per_channel: bool,
    static: bool,
    mode: quantizer.QuantizationMode,
    num_data: int = 8,
) -> onnx.ModelProto:
    if not static:
        raise Exception("Currently only supports static quantization.")

    _verify_not_quantized(model)

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


def _transform(
    transformers: List[Callable[[onnx.ModelProto], onnx.ModelProto]], model: onnx.ModelProto
) -> onnx.ModelProto:
    for transform in transformers:
        model = transform(model)
    return model


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


def _verify_not_quantized(model: onnx.ModelProto) -> None:
    # TODO also don't accept quantized operators in com.microsoft domain.
    for node in model.graph.node:
        op_type = node.op_type
        if op_type in [_DEQUANTIZE_LINEAR, _QUANTIZE_LINEAR, _Q_LINEAR_CONV, _Q_LINEAR_MAT_MUL]:
            raise AlreadyQuantizedError(op_type)
