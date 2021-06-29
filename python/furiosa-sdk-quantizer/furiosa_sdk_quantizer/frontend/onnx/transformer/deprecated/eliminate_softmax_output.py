import onnx

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class EliminateSoftmaxOutput(Transformer):
    """
    from: Softmax -> graph_output
    to: graph_output
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = self.remove_softmax_new_pattern(model)
        model = self.remove_softmax_old_pattern(model)
        return model

    def remove_softmax_new_pattern(self, model):
        nodes_by_output_name = {
            node_output: node for node in model.graph.node for node_output in node.output
        }
        vi_by_output_name = {vi.name: vi for vi in model.graph.value_info}
        outputs_by_output_name = {output.name: output for output in model.graph.output}

        optimized_nodes = []
        removed_nodes = []

        for node in model.graph.node:
            if node in removed_nodes:
                continue

            if node.op_type != "Transpose":
                optimized_nodes.append(node)
                continue

            if node.output[0] not in outputs_by_output_name.keys():
                optimized_nodes.append(node)
                continue

            prev_node = nodes_by_output_name[node.input[0]]
            if prev_node.op_type != "Softmax":
                optimized_nodes.append(node)
                continue

            pprev_node = nodes_by_output_name[prev_node.input[0]]
            if pprev_node.op_type != "Transpose":
                optimized_nodes.append(node)
                continue

            output_node = outputs_by_output_name[node.output[0]]
            removed_nodes.extend([node, prev_node, pprev_node])
            model.graph.output.remove(output_node)

            # Graph must have at least one graph output
            if not len(model.graph.output):
                ppprev_node = nodes_by_output_name[pprev_node.input[0]]
                model.graph.output.append(vi_by_output_name[ppprev_node.output[0]])

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

    def remove_softmax_old_pattern(self, model):
        nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        vi_by_output_name = {vi.name: vi for vi in model.graph.value_info}
        outputs_by_output_name = {output.name: output for output in model.graph.output}

        optimized_nodes = []
        removed_nodes = []

        for node in model.graph.node:
            if node.op_type != "Softmax":
                optimized_nodes.append(node)
                continue

            # Softmax must be graph output
            if node.output[0] not in outputs_by_output_name.keys():
                optimized_nodes.append(node)
                continue

            output_node = outputs_by_output_name[node.output[0]]
            removed_nodes.append(node)
            model.graph.output.remove(output_node)

            # Graph must have at least one graph output
            if not len(model.graph.output):
                prev_node = nodes_by_output_name[node.input[0]]
                model.graph.output.append(vi_by_output_name[prev_node.output[0]])

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
