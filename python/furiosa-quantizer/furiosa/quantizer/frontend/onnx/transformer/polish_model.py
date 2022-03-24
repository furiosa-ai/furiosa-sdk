from typing import Dict, List, Optional

import onnx
import onnxoptimizer

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.transformer.convert_2d_sum_to_add import Convert2dSumToAdd
from furiosa.quantizer.frontend.onnx.transformer.infer_squeeze_axes import InferSqueezeAxes
from furiosa.quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa.quantizer.interfaces.transformer import Transformer


class PolishModel(Transformer[onnx.ModelProto]):
    """
    Essential graph transformer/optimizers
    """

    def __init__(self, input_shapes: Optional[Dict[str, List[int]]] = None) -> None:
        super().__init__()
        self.input_shapes = input_shapes

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = utils.name_nodes(model)
        model = utils.make_initializer_name_unique(model)
        model = utils.fix_batch_size_as_one(model)

        model = onnxoptimizer.optimize(model, passes=["extract_constant_to_initializer"])
        model = Convert2dSumToAdd().transform(model)

        model = utils.fixed_point(
            model,
            [
                lambda model: InferenceShape(model).inference_shape(self.input_shapes),
                InferSqueezeAxes().transform,
            ],
        )

        return model
