import onnx
from onnx.helper import ModelProto
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.onnx_model import OnnxModel

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class BertOnnxModel(OnnxModel):
    def fuse_layer_normalization(self):
        fusion = FusionLayerNormalization(self)
        fusion.apply()


class FuseLayerNormalization(Transformer):
    """
    from:
        Input --> ReduceMean --> S --> Pow --> ReduceMean --> Add --> Sqrt --> D
              -----------------> ub -----------------------------------------> iv --> Mul --> Add Output
    to:
        LayerNormalization
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        orig_model = ModelProto()
        orig_model.CopyFrom(model)

        optimizer = BertOnnxModel(model)
        optimizer.fuse_layer_normalization()

        model = optimizer.model
        layer_norm_by_input_name = {
            node.input[0]: node for node in model.graph.node if node.op_type == 'LayerNormalization'
        }

        # nodes are not topologically sorted as a result of onnxruntime optimization
        sorted_nodes = []
        visited = 0
        for node in orig_model.graph.node:
            if node in model.graph.node:
                sorted_nodes.append(node)
                if node.output[0] in layer_norm_by_input_name.keys():
                    sorted_nodes.append(layer_norm_by_input_name[node.output[0]])
                visited += 1

        if not visited:
            sorted_nodes = model.graph.node

        model = utils.rebuild_model(model, sorted_nodes)
        check_model(model)

        return model
