import onnx

from quantizer.frontend.onnx.transformer.extract_constant_to_initializer import ExtractConstantToInitializer
from quantizer.frontend.onnx.transformer.convert_clip_attr_to_input import ConvertClipAttrToInput
from quantizer.frontend.onnx.transformer.convert_2d_sum_to_add import Convert2dSumToAdd
from quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from quantizer.frontend.onnx.transformer import utils
from quantizer.interfaces.transformer import Transformer


class PolishModel(Transformer[onnx.ModelProto]):
    """
    Essential graph transformer/optimizers
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = utils.name_nodes(model)
        model = utils.fix_batch_size_as_one(model)

        model = ExtractConstantToInitializer().transform(model)
        model = ConvertClipAttrToInput().transform(model)
        model = Convert2dSumToAdd().transform(model)
        model = InferenceShape(model).inference_shape()

        return model
