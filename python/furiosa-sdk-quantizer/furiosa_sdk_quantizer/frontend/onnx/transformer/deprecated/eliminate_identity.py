import onnx

from onnx.helper import make_model

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class EliminateIdentity(Transformer):
    """
    from: ArgMax -> graph output
    to: graph output
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        outputs_by_output_name = {output.name: output for output in model.graph.output}

        optimized_nodes = []
        removed_nodes = []

        # handle case where Identity occurs in the middle of graph
        for node in model.graph.node:
            if node.op_type == "Constant":
                continue

            # TODO need to ease assumption that node has only one input if necessary
            try:
                prev_node = nodes_by_output_name[node.input[0]]
            except KeyError:
                continue

            if prev_node.op_type != "Identity":
                continue

            node.input[0] = prev_node.input[0]
            removed_nodes.append(prev_node)

        # handle case where Identity occurs at the end of graph
        for node in model.graph.node:
            if node.op_type != "Identity":
                optimized_nodes.append(node)
                continue

            # Identity must be a graph output
            try:
                output_node = outputs_by_output_name[node.output[0]]
                model.graph.output.remove(output_node)
            except KeyError:
                continue

            removed_nodes.append(node)
            # Graph must have at least one graph output
            prev_node = nodes_by_output_name[node.input[0]]

            new_output_node = outputs_by_output_name[node.output[0]]
            new_output_node.name = prev_node.output[0]
            model.graph.output.append(new_output_node)

        # remove duplicate node(s) in optimized nodes
        seen = []
        for op_node in optimized_nodes:
            if op_node in seen:
                continue
            seen.append(op_node)
        optimized_nodes = seen

        new_nodes = list(filter(lambda node: node not in removed_nodes, optimized_nodes))
        model = utils.rebuild_model(model, new_nodes)
        check_model(model)

        return model
