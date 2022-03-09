import onnx
from onnx import version_converter

from furiosa.quantizer.frontend.onnx import __OPSET_VERSION__
from furiosa.quantizer.frontend.onnx.transformer.utils import include_initializer_to_graph_input
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class CheckVersion(Transformer[onnx.ModelProto]):
    """Convert an ONNX model to ONNX opset 12"""

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # https://github.com/onnx/onnx/issues/2873#issuecomment-652541006
        #
        # > There is an underlying issue in version converter. It relies on the
        # > C++ IR which I believe has not been updated after IR v3. Because of
        # > this I think it expects initializers also be added as graph inputs.
        # > If you try to change the version of your model to IRv3 or create a
        # > model with initializers also as inputs then I think this will work.
        model = include_initializer_to_graph_input(model)

        version = int(model.opset_import[0].version)
        if version != __OPSET_VERSION__:
            try:
                model = version_converter.convert_version(model, __OPSET_VERSION__)
                check_model(model)
            except Exception as exc:
                raise NotImplementedError(
                    f"can't convert the model (ONNX opset {version}) to ONNX opset {__OPSET_VERSION__}"
                ) from exc
        return model
