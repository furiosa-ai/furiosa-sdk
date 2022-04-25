import copy
from typing import List

import onnx

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class EliminateSSDDetectionPostprocess(Transformer):
    """
    from: Softmax -> graph_output
    to: graph_output

    Assume NCHW Input
    """

    def __init__(self, ssd_outputs: List):
        self.ssd_outputs = ssd_outputs

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        vi_by_names = {vi.name: vi for vi in model.graph.value_info}

        removed_nodes = _get_postprocess_nodes(model, self.ssd_outputs)

        new_nodes = [node for node in model.graph.node if node not in removed_nodes]
        model = utils.rebuild_model(model, new_nodes)
        for output in self.ssd_outputs:
            model.graph.output.append(vi_by_names[output])
        check_model(model)

        return model


def _get_postprocess_nodes(model, ssd_output_tensors):
    inputs = set()
    inputs.update(ssd_output_tensors)

    postprocess_nodes = []

    # forward traverse
    for node in model.graph.node:
        is_append = False
        for node_input in node.input:
            if node_input in inputs:
                is_append = True
                for node_output in node.output:
                    inputs.add(node_output)

        if is_append:
            postprocess_nodes.append(node)

    # backward traverse
    while True:
        prev_postprocess_nodes = copy.deepcopy(postprocess_nodes)

        for node in model.graph.node:
            is_append = False
            for postprocess_node in postprocess_nodes:
                for node_output in node.output:
                    if (
                        node_output in postprocess_node.input
                        and node_output not in ssd_output_tensors
                    ):
                        is_append = True
            if is_append and node not in postprocess_nodes:
                postprocess_nodes.append(node)

        if len(prev_postprocess_nodes) == len(postprocess_nodes):
            break

    return postprocess_nodes
