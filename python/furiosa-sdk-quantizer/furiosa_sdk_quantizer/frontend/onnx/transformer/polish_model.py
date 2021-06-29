import onnx

from furiosa_sdk_quantizer.frontend.onnx.transformer.extract_constant_to_initializer import (
    ExtractConstantToInitializer,
)
from furiosa_sdk_quantizer.frontend.onnx.transformer.convert_clip_attr_to_input import (
    ConvertClipAttrToInput,
)
from furiosa_sdk_quantizer.frontend.onnx.transformer.convert_2d_sum_to_add import Convert2dSumToAdd
from furiosa_sdk_quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.interfaces.transformer import Transformer


class PolishModel(Transformer[onnx.ModelProto]):
    """
    Essential graph transformer/optimizers
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = utils.name_nodes(model)
        model = utils.make_conv_bias_name_unique(model)
        model = utils.fix_batch_size_as_one(model)

        model = ExtractConstantToInitializer().transform(model)
        model = ConvertClipAttrToInput().transform(model)
        model = Convert2dSumToAdd().transform(model)
        model = InferenceShape(model).inference_shape()

        return model
