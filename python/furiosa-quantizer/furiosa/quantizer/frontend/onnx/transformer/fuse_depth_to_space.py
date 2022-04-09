from typing import Dict, Iterable

import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseDepthToSpace(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Reshape --> Transpose --> Reshape --> next
    to
        prev --> DepthToSpace --> next

    if Transpose.perm == [0, 1, 4, 2, 5, 3] or == [0, 3, 4, 1, 5, 2]
    """

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
        pattern_to_match = ['Reshape', 'Transpose', 'Reshape']
        matched_nodes = self.pattern_matcher(base_node, pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not _pattern_condition_checker(matched_nodes):
            return base_node.input

        reshape, *_ = matched_nodes
        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[
                onnx.helper.make_node(
                    'DepthToSpace',
                    [reshape.input[0]],
                    [base_node.output[0]],
                    reshape.name,
                    **self.get_attrs(matched_nodes),
                )
            ],
        )

        return reshape.input

    def get_attrs(self, matched_nodes: Iterable[onnx.NodeProto]) -> Dict:
        reshape, transpose, _ = matched_nodes
        permutation = next(
            onnx.helper.get_attribute_value(attr)
            for attr in transpose.attribute
            if attr.name == "perm"
        )
        shape_to_reshape = self.get_value_info_shape(reshape.output[0])
        if permutation == [0, 1, 4, 2, 5, 3]:
            mode = 'CRD'
            blocksize = shape_to_reshape[2]
        elif permutation == [0, 3, 4, 1, 5, 2]:
            mode = 'DCR'
            blocksize = shape_to_reshape[1]
        else:
            assert False, repr(permutation)
        return {'blocksize': blocksize, 'mode': mode}


def _pattern_condition_checker(nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
    _, transpose, _ = nodes_to_check
    perm = next(
        onnx.helper.get_attribute_value(attr) for attr in transpose.attribute if attr.name == "perm"
    )
    return perm in ([0, 1, 4, 2, 5, 3], [0, 3, 4, 1, 5, 2])
