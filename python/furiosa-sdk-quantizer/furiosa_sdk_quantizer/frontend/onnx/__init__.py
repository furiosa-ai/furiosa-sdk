from typing import Dict, List, Tuple, Callable, Text, IO, Optional

import onnx

__DOMAIN__ = ''
__OPSET_VERSION__ = 12

from furiosa_sdk_quantizer.frontend.onnx import spec
from furiosa_sdk_quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa_sdk_quantizer.frontend.onnx.utils.version_checker import CheckVersion
from furiosa_sdk_quantizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa_sdk_quantizer.frontend.onnx.transformer.eliminate_argmax_output import EliminateArgmaxOutput
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_bn_into_conv import FuseBnIntoConv
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_lp_normalization import FuseLpNormalization
from furiosa_sdk_quantizer.frontend.onnx.transformer.deprecated.fuse_scalar_mul_into_conv import FuseScalarMulIntoConv
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_conv import FuseConv
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_depth_to_space import FuseDepthToSpace
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_gelu import FuseGELU
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_layer_normalization import FuseLayerNormalization
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_redundant_reshape_pattern import FuseRedundantReshapePattern
from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_pad import FusePad
from furiosa_sdk_quantizer.frontend.onnx.transformer.eliminate_redundant_reshape_pattern import EliminateRedundantReshapePattern
from furiosa_sdk_quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import ConvertConv1dToConv2d
from furiosa_sdk_quantizer.frontend.onnx.quantizer.calibrator import ONNXCalibrator
from furiosa_sdk_quantizer.frontend.onnx.quantizer import quantizer


def _transform(transformers: List[Callable[[onnx.ModelProto], onnx.ModelProto]],
               model: onnx.ModelProto) -> onnx.ModelProto:
    for transform in transformers:
        model = transform(model)
    return model


def _inference_shape(model: onnx.ModelProto) -> onnx.ModelProto:
    return _transform([
        PolishModel().transform,
    ], model)


def _reify(model: onnx.ModelProto) -> onnx.ModelProto:
    transformers = [
        ConvertConv1dToConv2d().transform,
        FuseConv().transform,
        FusePad().transform,
        FuseBnIntoConv().transform,
        FuseDepthToSpace().transform,
        FuseGELU().transform,
        FuseLayerNormalization().transform,
        FuseLpNormalization().transform,
        FuseRedundantReshapePattern().transform,
        EliminateArgmaxOutput().transform,
        EliminateRedundantReshapePattern().transform,
    ]
    return _transform(transformers, model)


def export_spec(model: onnx.ModelProto, output: IO[Text]):
    model = _transform([_inference_shape, _reify], model)
    spec.export_spec.OnnxExportSpec(model).dump(output)


def optimize_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = _transform([CheckVersion().transform], model)

    # TODO hotfix if _inference_shape is not applied
    if any(vi_name not in model.graph.value_info for node in model.graph.node for vi_name in node.output):
        model = _transform([_inference_shape, _reify], model)

    # TODO coldfix if graph_transformation is not applied

    return model


def build_calibration_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = optimize_model(model)
    return ONNXCalibrator(model).build_calibration_model()


def quantize(model: onnx.ModelProto,
             per_channel: bool,
             static: bool,
             mode: quantizer.QuantizationMode,
             dynamic_ranges: Dict[str, Tuple[float, float]]) -> onnx.ModelProto:
    return quantizer.FuriosaONNXQuantizer(model,
                                          per_channel,
                                          static,
                                          mode,
                                          dynamic_ranges).quantize()


def post_training_quantization_with_random_calibration(model: onnx.ModelProto,
                                                       per_channel: bool,
                                                       static: bool,
                                                       mode: quantizer.QuantizationMode,
                                                       num_data: Optional[int] = None) -> onnx.ModelProto:
    if not static:
        raise Exception("Currently only supports static quantization.")
    if mode not in [quantizer.QuantizationMode.dfg, quantizer.QuantizationMode.fake]:
        raise Exception("Currently only supports QuantizationMode dfg or fake.")

    model = optimize_model(model)
    calibration_model = build_calibration_model(model)
    dynamic_ranges = ONNXCalibrator(calibration_model).calibrate_with_random(num_data)
    return quantize(model, per_channel, static, mode, dynamic_ranges)


def calibrate_with_random(model: onnx.ModelProto, num_data: Optional[int] = None) -> Dict[str, Tuple[float, float]]:
    model = optimize_model(model)
    calibration_model = ONNXCalibrator(model).build_calibration_model()
    return ONNXCalibrator(calibration_model).calibrate_with_random(num_data)
