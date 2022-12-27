from typing import Iterable, List

import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseRedundantReshapePattern(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_2,
            Pattern_1,
            Pattern_3,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Reshape --> Reshape --> next
    to
        prev --> Reshape --> next

    if prev.output[0].shape != next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Reshape']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
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
        reshape = matched_nodes[0]
        return reshape.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        reshape, reshape1 = nodes_to_check
        return not self.is_same_shape(reshape.input[0], reshape1.output[0])

    @staticmethod
    def make_new_node(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        reshape, reshape1 = matched_nodes
        return [
            onnx.helper.make_node(
                'Reshape',
                [reshape.input[0], reshape.input[1] + '_fused'],
                [reshape1.output[0]],
                name=reshape.name,
            )
        ]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        reshape, reshape1 = matched_nodes
        return [
            onnx.helper.make_tensor(
                name=reshape.input[1] + '_fused',
                data_type=onnx.TensorProto.INT64,
                dims=[
                    len(self.get_value_info_shape(reshape1.output[0])),
                ],
                vals=self.get_value_info_shape(reshape1.output[0]),
            )
        ]


class Pattern_2(ONNXTransformer):
    """
    transform
        prev --> Reshape --> Reshape --> Reshape --> next
    to
        prev --> Reshape --> next

    if prev.output[0].shape != next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Reshape', 'Reshape']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
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
        reshape = matched_nodes[0]
        return reshape.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        reshape, _, reshape2 = nodes_to_check
        return not self.is_same_shape(reshape.input[0], reshape2.output[0])

    @staticmethod
    def make_new_node(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        reshape, _, reshape2 = matched_nodes
        return [
            onnx.helper.make_node(
                'Reshape',
                [reshape.input[0], reshape.input[1] + '_fused'],
                [reshape2.output[0]],
                name=reshape.name,
            )
        ]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        reshape, _, reshape2 = matched_nodes
        return [
            onnx.helper.make_tensor(
                name=reshape.input[1] + '_fused',
                data_type=onnx.TensorProto.INT64,
                dims=[
                    len(self.get_value_info_shape(reshape2.output[0])),
                ],
                vals=self.get_value_info_shape(reshape2.output[0]),
            )
        ]


class Pattern_3(ONNXTransformer):
    """
    transform
        prev --> Flatten/Squeeze --> Unsqueeze --> next
    to
        prev --> Reshape --> next
    if prev.output[0].shape != next.input[0].shape
    """

    pattern_to_match = ['Flatten/Squeeze', 'Unsqueeze']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
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
        flatten = matched_nodes[0]
        return flatten.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        flatten, unsqueeze = nodes_to_check
        return not self.is_same_shape(flatten.input[0], unsqueeze.output[0])

    @staticmethod
    def make_new_node(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        flatten, unsqueeze = matched_nodes
        return [
            onnx.helper.make_node(
                'Reshape',
                [flatten.input[0], flatten.input[0] + '_fused'],
                [unsqueeze.output[0]],
                flatten.name,
            )
        ]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        flatten, unsqueeze = matched_nodes
        return [
            onnx.helper.make_tensor(
                name=flatten.input[0] + '_fused',
                data_type=onnx.TensorProto.INT64,
                dims=[
                    len(self.get_value_info_shape(unsqueeze.output[0])),
                ],
                vals=self.get_value_info_shape(unsqueeze.output[0]),
            )
        ]
