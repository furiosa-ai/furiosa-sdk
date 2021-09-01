import onnx
import onnxoptimizer

from furiosa.quantizer.frontend.onnx.transformer.convert_2d_sum_to_add import Convert2dSumToAdd
from furiosa.quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.interfaces.transformer import Transformer


class PolishModel(Transformer[onnx.ModelProto]):
    """
    Essential graph transformer/optimizers
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = utils.name_nodes(model)
        model = utils.make_conv_bias_name_unique(model)
        model = utils.fix_batch_size_as_one(model)

        model = onnxoptimizer.optimize(model, passes=["extract_constant_to_initializer"])
        model = Convert2dSumToAdd().transform(model)
        model = InferenceShape(model).inference_shape()

        return model
