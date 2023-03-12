from typing import Optional

import onnx
from onnx import version_converter

from furiosa.optimizer.frontend.onnx import __OPSET_VERSION__
from furiosa.optimizer.frontend.onnx.utils.check_model import check_model
from furiosa.optimizer.interfaces.transformer import Transformer


class CheckVersion(Transformer[onnx.ModelProto]):  # pylint: disable=no-member
    """Convert an ONNX model to ONNX opset 12 or 13"""

    opset_bound = [12, 13]

    def __init__(self, opset_version: Optional[int] = None) -> None:
        if opset_version < self.opset_bound[0] or opset_version > self.opset_bound[1]:
            raise ValueError(
                f"Unsupported opset_version: {opset_version}. Choose {self.opset_bound[0]} <= opset_version <= {self.opset_bound[1]}."
            )
        if opset_version is None:
            self.opset_version = __OPSET_VERSION__
        self.opset_version = opset_version

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:  # pylint: disable=no-member
        version = int(model.opset_import[0].version)
        if version != self.opset_version:
            try:
                model = version_converter.convert_version(model, self.opset_version)
                check_model(model, check_runnable=False)
            except Exception as exc:
                raise NotImplementedError(
                    f"can't convert the model (ONNX opset {version}) to ONNX opset {__OPSET_VERSION__}"
                ) from exc
        return model
