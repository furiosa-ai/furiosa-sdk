from typing import Dict, List, Optional

import onnx
import onnxsim

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model


class InferenceShape:
    """
    Replace former InferenceShape with ONNX_Simplifier
    https://github.com/daquexian/onnx-simplifier
    """

    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = utils.rebuild_model(model, list(model.graph.node))

    def inference_shape(
        self, input_shapes: Optional[Dict[str, List[int]]] = None
    ) -> onnx.ModelProto:
        try:
            self.model, check = onnxsim.simplify(
                self.model,
                skipped_optimizers=['eliminate_duplicate_initializer', 'fuse_add_bias_into_conv'],
                input_shapes=input_shapes,
            )

        except RuntimeError as e:
            import re  # pylint: disable=import-outside-toplevel

            # https://github.com/daquexian/onnx-simplifier/blob/99f544e805d952ff35e682ec276c8d3cd1a377e5/onnxsim/onnx_simplifier.py#L98-L102
            m = re.match(r'The shape of input "(\w+)" has dynamic size', e.args[0])
            if m is not None:
                raise RuntimeError(
                    f'the static shape of input "{m.group(1)}" must be given'
                ) from None
            raise

        assert check
        check_model(self.model)

        return utils.name_nodes(self.model)
