import onnx

import onnxsim
from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model


class InferenceShape:
    """
        Replace former InferenceShape with ONNX_Simplifier
        https://github.com/daquexian/onnx-simplifier
    """

    def __init__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        self.model = utils.rebuild_model(model, model.graph.node)

    def inference_shape(self) -> onnx.ModelProto:
        self.model, check = onnxsim.simplify(self.model, skipped_optimizers=['eliminate_duplicate_initializer'])
        assert check
        check_model(self.model)

        return utils.name_nodes(self.model)
