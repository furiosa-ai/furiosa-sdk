from typing import Iterable, List

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

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)

        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
        )

        gather, _ = matched_nodes
        return gather.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        _, matmul = nodes_to_check
        cond1 = len(self.get_value_info_shape(matmul.output[0])) == 2
        cond2 = (matmul.input[0] in self.initializer_map) != (
            matmul.input[1] in self.initializer_map
        )
        return cond1 and cond2

    @staticmethod
    def make_new_node(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        gather, matmul = matched_nodes

        return [
            onnx.helper.make_node(
                'Gather',
                inputs=[gather.input[0] + '_fused', gather.input[1]],
                outputs=[matmul.output[0]],
                name=matmul.output[0] + '_1',
            )
        ]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        gather, matmul = matched_nodes

        table_tensor = gather.input[0]
        weight_tensor = self.get_init_node_input(matmul)

        table_arr = self.get_initializer_array(table_tensor)
        weight_arr = self.get_initializer_array(weight_tensor)

        fused_table_arr = np.matmul(table_arr, weight_arr)
        return [onnx.numpy_helper.from_array(fused_table_arr, table_tensor + '_fused')]
