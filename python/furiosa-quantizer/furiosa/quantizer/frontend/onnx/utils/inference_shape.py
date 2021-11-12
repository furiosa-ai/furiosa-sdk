from typing import Optional, Dict, List

import onnx
import onnxsim

from onnxsim.onnx_simplifier import get_input_names
from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model


class InferenceShape:
    """
    Replace former InferenceShape with ONNX_Simplifier
    https://github.com/daquexian/onnx-simplifier
    """

    def __init__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        self.model = utils.rebuild_model(model, model.graph.node)

    def inference_shape(
        self, input_shapes: Optional[Dict[str, List[int]]] = None
    ) -> onnx.ModelProto:
        try:
            self.model, check = onnxsim.simplify(
                self.model,
                skipped_optimizers=['eliminate_duplicate_initializer', 'fuse_add_bias_into_conv'],
                input_shapes=input_shapes,
            )
        except RuntimeError:
            input_names = get_input_names(self.model)
            raise RuntimeError(f"Shape(s) must be specified for graph input(s): {input_names}")

        assert check
        check_model(self.model)

        return utils.name_nodes(self.model)
