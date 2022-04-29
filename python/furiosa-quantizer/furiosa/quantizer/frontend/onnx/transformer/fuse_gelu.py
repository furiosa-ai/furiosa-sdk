import onnx
from onnx.helper import ModelProto
from onnxruntime.transformers.fusion_gelu import FusionGelu
from onnxruntime.transformers.onnx_model import OnnxModel

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class BertOnnxModel(OnnxModel):
    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()


class FuseGELU(Transformer):
    """
    from:
        Input --> Div --> Erf --> Add --> M
              ------------------> Mul --> ul--> Output
    to:
        GELU
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        orig_model = ModelProto()
        orig_model.CopyFrom(model)

        optimizer = BertOnnxModel(model)
        optimizer.fuse_gelu()

        model = optimizer.model
        gelu_by_input_name = {
            node.input[0]: node for node in model.graph.node if node.op_type == 'Gelu'
        }

        # nodes are not topologically sorted as a result of onnxruntime optimization
        sorted_nodes = []
        visited = 0
        for node in orig_model.graph.node:
            if node in model.graph.node:
                sorted_nodes.append(node)
                if node.output[0] in gelu_by_input_name:
                    sorted_nodes.append(gelu_by_input_name[node.output[0]])
                visited += 1

        if not visited:
            sorted_nodes = list(model.graph.node)

        model = utils.rebuild_model(model, sorted_nodes)
        check_model(model)

        return model
