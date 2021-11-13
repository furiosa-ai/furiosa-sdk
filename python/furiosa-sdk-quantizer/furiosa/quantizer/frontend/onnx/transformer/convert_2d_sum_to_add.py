import onnx
from onnx.helper import TensorProto, make_node, make_tensor

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class Convert2dSumToAdd(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        optimized_nodes = []
        for node in model.graph.node:
            if node.op_type != 'Sum':
                optimized_nodes.append(node)
                continue

            if len(node.input) != 2:
                optimized_nodes.append(node)
                continue

            new_node = make_node(
                'Add', inputs=[node.input[0], node.input[1]], outputs=[node.output[0]]
            )

            optimized_nodes.append(new_node)

        # remove duplicate node(s) in optimized nodes
        seen = []
        for op_node in optimized_nodes:
            if op_node in seen:
                continue
            seen.append(op_node)
        optimized_nodes = seen

        model = utils.rebuild_model(model, optimized_nodes)
        check_model(model)

        return model
