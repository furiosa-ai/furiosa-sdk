from typing import Iterable, List

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx.transformer import ONNXTransformer, utils
from furiosa.optimizer.interfaces.transformer import Transformer


class FuseGatherMatMul(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:  # pylint: disable=no-member
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

    if 1. MatMul must have exactly one initializer
       2. Gather.data must be defined in graph.initializer
       3. MatMul weight's data_type == onnx.TensorProto.FLOAT
       4. rank(MatMul weight) == 2
       5. Gather.data's data_type == onnx.TensorProto.FLOAT
       6. rank(Gather.data) == 2
       7. (Gather.axis == 0 and Matmul.input[1] is initializer) or (Gather.axis == 1 and Matmul.input[0] is initializer)
    """

    pattern_to_match = ['Gather', 'MatMul']

    def pattern_matching(
        self, base_node: onnx.NodeProto  # pylint: disable=no-member
    ) -> Iterable[str]:
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

    # pylint: disable=too-many-return-statements
    def pattern_condition_checker(
        self, nodes_to_check: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> bool:
        gather, matmul = nodes_to_check

        if (matmul.input[0] in self.initializer_map) == (matmul.input[1] in self.initializer_map):
            return False

        matmul_weight = self.get_init_node_input(matmul)

        gather_data = gather.input[0]
        if gather_data not in self.initializer_map:
            return False

        # pylint: disable-next=no-member
        if self.get_value_info_dtype(matmul_weight) != onnx.TensorProto.FLOAT:
            return False

        if len(self.get_value_info_shape(matmul_weight)) != 2:
            return False

        # pylint: disable-next=no-member
        if self.get_value_info_dtype(gather_data) != onnx.TensorProto.FLOAT:
            return False

        if len(self.get_value_info_shape(gather_data)) != 2:
            return False

        gather_axis = utils.get_attribute(gather.attribute, "axis", default=0)
        if not (
            (gather_axis == 0 and matmul.input[1] in self.initializer_map)
            or (gather_axis == 1 and matmul.input[0] in self.initializer_map)
        ):
            return False
        return True

    @staticmethod
    def make_new_node(
        matched_nodes: Iterable[onnx.NodeProto],  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        gather, matmul = matched_nodes

        return [
            onnx.helper.make_node(
                'Gather',
                inputs=[gather.input[0] + '_fused', gather.input[1]],
                outputs=[matmul.output[0]],
                name=matmul.output[0] + '_1',
                **utils.get_node_attributes(gather),
            )
        ]

    def make_new_init(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.TensorProto]:  # pylint: disable=no-member
        gather, matmul = matched_nodes

        table_tensor = gather.input[0]
        weight_tensor = self.get_init_node_input(matmul)

        table_arr = self.get_initializer_array(table_tensor)
        weight_arr = self.get_initializer_array(weight_tensor)

        init_idx = list(matmul.input).index(weight_tensor)
        if init_idx == 1:
            fused_table_arr = np.matmul(table_arr, weight_arr)
        elif init_idx == 0:
            fused_table_arr = np.matmul(weight_arr, table_arr)

        return [onnx.numpy_helper.from_array(fused_table_arr, table_tensor + '_fused')]
