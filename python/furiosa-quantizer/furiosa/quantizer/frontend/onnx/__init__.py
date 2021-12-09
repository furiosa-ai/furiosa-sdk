from typing import IO, Callable, Dict, List, Optional, Text, Tuple

import numpy as np
import onnx

__DOMAIN__ = ''
__OPSET_VERSION__ = 12

from furiosa.quantizer.frontend.onnx import calibrate, spec
from furiosa.quantizer.frontend.onnx.quantizer import quantizer
from furiosa.quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import (
    ConvertConv1dToConv2d,
)
from furiosa.quantizer.frontend.onnx.transformer.eliminate_redundant_shape_pattern import (
    EliminateRedundantShapePattern,
)
from furiosa.quantizer.frontend.onnx.transformer.fuse_bn_into_conv import FuseBnIntoConv
from furiosa.quantizer.frontend.onnx.transformer.fuse_bn_into_convtranspose import (
    FuseBnIntoConvTranspose,
)
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
        FuseBnIntoConvTranspose().transform,
        FuseBnIntoConv().transform,
        FuseDepthToSpace().transform,
        FuseGELU().transform,
        FuseLayerNormalization().transform,
        FuseLpNormalization().transform,
        FuseRedundantReshapePattern().transform,
        FuseGatherMatMul().transform,
        EliminateRedundantShapePattern().transform,
    ]
    return _transform(transformers, model)


def export_spec(model: onnx.ModelProto, output: IO[Text]):
    model = _transform([_inference_shape, _reify], model)
    spec.export_spec.OnnxExportSpec(model).dump(output)


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
    model = optimize_model(model)
    ranges = calibrate.calibrate(model, dataset)
    return quantize(model, per_channel, True, quantizer.QuantizationMode.dfg, ranges)


def post_training_quantization_with_random_calibration(
    model: onnx.ModelProto,
    per_channel: bool,
    static: bool,
    mode: quantizer.QuantizationMode,
    num_data: int = 8,
) -> onnx.ModelProto:
    if not static:
        raise Exception("Currently only supports static quantization.")
    if mode not in [quantizer.QuantizationMode.dfg, quantizer.QuantizationMode.fake]:
        raise Exception("Currently only supports QuantizationMode dfg or fake.")

    model = optimize_model(model)
    dynamic_ranges = calibrate.calibrate_with_random_data(model, num_data)
    return quantize(model, per_channel, static, mode, dynamic_ranges)
