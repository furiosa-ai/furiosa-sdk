from typing import Dict, Iterable

import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.frontend.onnx.transformer.utils import get_attribute
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseLpNormalization(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> ReduceL2/ReduceL1 --> Clip --> Expand -->  Div --> next
             +                                           +
             -------------------------------------------->
    to
        prev --> LpNormalization --> next
    # TODO Check if Div has no initialzier
    """

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
        inputs = base_node.input

        pattern_to_match = ['ReduceL2/ReduceL1', 'Clip', 'Expand', 'Div']
        matched_nodes = self.pattern_matcher(base_node, pattern_to_match)
        if not matched_nodes:
            return inputs

        reduce_lp, *_ = matched_nodes
        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[
                onnx.helper.make_node(
                    'LpNormalization',
                    [reduce_lp.input[0]],
                    [base_node.output[0]],
                    reduce_lp.name,
                    **_get_attrs(reduce_lp),
                )
            ],
        )

        return reduce_lp.input


# TODO Implement Pattern_2 in case of unsimplified graph, containing Shape operator:
# transform
#   prev --> ReduceL2/ReduceL1 --> Clip --> Expand -->  Div --> next
#        +                                +          +
#        ------------------------> Shape ->
#        +                                           +
#        -------------------------------------------->
# to
#   prev --> LpNormalization --> next


def _get_attrs(node: onnx.NodeProto) -> Dict:
    assert node.op_type in ("ReduceL1", "ReduceL2")
    axes = get_attribute(node.attribute, "axes")
    if node.op_type == 'ReduceL1':
        p = 1
    elif node.op_type == 'ReduceL2':
        p = 2
    return {"axis": int(axes[0]), "p": p}
