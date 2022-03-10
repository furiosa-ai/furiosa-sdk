import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseGatherMatMul(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Gather --> MatMul --> next
    to
        prev --> Gather --> next

    if 1. MatMul.ndim == 2
       2. MatMul must have exactly one initializer
    """

    pattern_to_match = ['Gather', 'MatMul']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)

        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[_make_nodes(matched_nodes)],
            inits_to_add=[self.make_initializers(*self.get_new_init_args(matched_nodes))],
        )
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        _, base_node = nodes_to_check

        return self.check_condition_1(base_node.output[0]) and self.check_condition_2(base_node)

    def check_condition_1(self, tensor_name):
        return len(self.get_value_info_shape(tensor_name)) == 2

    def check_condition_2(self, node):
        return sum(node_input in self.initializer_map for node_input in node.input) == 1

    def get_new_init_args(self, matched_nodes):
        top_node, base_node = matched_nodes

        table_input = self.get_init_node_input(top_node)
        matmul_input = self.get_init_node_input(base_node)

        return table_input, matmul_input

    def make_initializers(self, top_node_init, base_node_init):
        table_arr = self.get_initializer_array(top_node_init)
        matmul_arr = self.get_initializer_array(base_node_init)

        new_table_arr = np.matmul(table_arr, matmul_arr)

        new_init = onnx.numpy_helper.from_array(new_table_arr, top_node_init + '_fused')

        return new_init


def _make_nodes(matched_nodes):
    top_node, base_node = matched_nodes

    fused_gather_node = onnx.helper.make_node(
        'Gather',
        inputs=[top_node.input[0] + '_fused', top_node.input[1]],
        outputs=[base_node.output[0]],
        name=base_node.output[0] + '_1',
    )

    return fused_gather_node
