import onnx
from onnx import version_converter

from furiosa.quantizer.frontend.onnx import __OPSET_VERSION__
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class CheckVersion(Transformer[onnx.ModelProto]):
    """Convert an ONNX model into furiosa.quantizer.frontend.onnx.__OPSET_VERSION__"""

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        version = int(model.opset_import[0].version)
        if version != __OPSET_VERSION__:
            try:
                model = version_converter.convert_version(model, __OPSET_VERSION__)
                check_model(model, check_runnable=False)
            except Exception as exc:
                raise NotImplementedError(
                    f"can't convert the model (ONNX opset {version}) to ONNX opset {__OPSET_VERSION__}"
                ) from exc
        return model
