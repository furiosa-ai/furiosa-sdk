import abc

import onnx

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import ONNXTransformer


class FuseLpNormalization(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer, abc.ABC):
    """
    transform
        prev --> ReduceL2/ReduceL1 --> Clip --> Expand -->  Div --> next
             +                                           +
             -------------------------------------------->
    to
        prev --> LpNormalization --> next
    """

    def pattern_matching(self, base_node):
        inputs = base_node.input

        pattern_to_match = ["ReduceL2/ReduceL1", "Clip", "Expand", "Div"]
        matched_nodes = self.pattern_matcher(base_node, pattern_to_match)
        if not matched_nodes:
            return inputs

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[
                self.make_node(
                    "LpNormalization",
                    [top_node.input[0]],
                    [base_node.output[0]],
                    top_node.name,
                    **self.get_attrs(top_node)
                )
            ],
        )

        return top_node.input

    def get_attrs(self, node):
        from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import attribute_to_kwargs

        attrs = attribute_to_kwargs(node.attribute)

        if node.op_type == "ReduceL1":
            p = 1
        elif node.op_type == "ReduceL2":
            p = 2
        else:
            raise Exception()

        return {"axis": int(attrs["axes"][0]), "p": int(p)}
