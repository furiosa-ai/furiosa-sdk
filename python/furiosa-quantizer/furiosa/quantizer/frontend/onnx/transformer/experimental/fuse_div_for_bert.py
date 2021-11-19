import onnx
from onnx import numpy_helper
from onnx.helper import make_model

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseDivForBert(Transformer):
    """
    Only works for some BERT Models
    """

    def __init__(self):
        self.nodes_by_output_name = None
        self.initializers = None
        self.outputs_by_name = None

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = PolishModel().transform(model)

        self.nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        self.initializers = {init.name: init for init in model.graph.initializer}
        self.outputs_by_name = {oup.name: oup for oup in model.graph.output}

        model = self.transform_matmul_add(model)  # transform matmul + add --> conv
        check_model(model)

        return PolishModel().transform(model)

    def transform_matmul_add(self, model):
        optimized_nodes = []
        removed_nodes = []

        # Handle Case1: MatMul + Add
        for node in model.graph.node:
            if node.op_type != 'Div':
                optimized_nodes.append(node)
                continue

            prev_node_1 = self.nodes_by_output_name[node.input[0]]
            if prev_node_1.op_type != 'MatMul':
                optimized_nodes.append(node)
                continue

            prev_node_2 = self.nodes_by_output_name[prev_node_1.input[0]]
            prev_node_3 = self.nodes_by_output_name[prev_node_2.input[0]]
            prev_node_4 = self.nodes_by_output_name[prev_node_3.input[0]]

            assert prev_node_4.op_type == 'Add'

            scalar = numpy_helper.to_array(self.initializers[node.input[1]])
            arr = numpy_helper.to_array(self.initializers[prev_node_4.input[1]])

            model.graph.initializer.append(
                numpy_helper.from_array(arr / scalar, name=prev_node_4.input[1] + '_div_fused')
            )
            prev_node_4.input[1] = prev_node_4.input[1] + '_div_fused'
            prev_node_1.output[0] = node.output[0]
            removed_nodes.append(node)

        new_nodes = list(filter(lambda node: node not in removed_nodes, optimized_nodes))
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        model = make_model(model.graph)

        model = utils.eliminate_unused_protos(model)

        return model
