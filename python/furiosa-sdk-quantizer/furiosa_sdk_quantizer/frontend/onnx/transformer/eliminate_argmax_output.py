import onnx
import abc

from furiosa_sdk_quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa_sdk_quantizer.interfaces.transformer import Transformer


class EliminateArgmaxOutput(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer, abc.ABC):
    """
    transform
        prev --> Argmax --> next
    to
        prev --> (   ) --> next
    if next is one of graph outputs
    """

    def pattern_matching(self, base_node):
        inputs = base_node.input

        pattern_to_match = ["ArgMax"]
        matched_nodes = self.pattern_matcher(base_node, pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        self.transform_to_eliminate(matched_nodes, base_node.input[0])
        return inputs

    def pattern_condition_checker(self, nodes_to_check):
        node = nodes_to_check[0]
        if any(node.output[0] == output for output in self.graph_output_map):
            return True
        return False
