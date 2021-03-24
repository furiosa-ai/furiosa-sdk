import onnx

from onnx import version_converter

from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model
from furiosa_sdk_quantizer.interfaces.transformer import Transformer


class CheckVersion(Transformer[onnx.ModelProto]):
    """
    Version checker
    Convert version < 9 to 9
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        version = int(model.opset_import[0].version)

        try:
            if version <= 8:
                model = version_converter.convert_version(model, 9)
        except Exception as e:
            raise Exception(f"Failed to convert onnx from {version} to 9, the quantizer officially supports 12, {e}")

        return model
