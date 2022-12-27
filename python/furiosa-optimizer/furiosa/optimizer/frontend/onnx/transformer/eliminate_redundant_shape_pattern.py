import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class EliminateRedundantShapePattern(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
            Pattern_4,
            Pattern_5,
            Pattern_6,
            Pattern_7,
            Pattern_3,
            Pattern_8,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Flatten/Squeeze --> Unsqueeze --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Flatten/Squeeze', 'Unsqueeze']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node = matched_nodes[0]

        self.transform_to_eliminate(matched_nodes, top_node.input[0])
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        top_node = nodes_to_check[0]
        base_node = nodes_to_check[-1]

        if self.is_same_shape(top_node.input[0], base_node.output[0]):
            return True
        return False


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Reshape --> Flatten/Squeeze --> Unsqueeze --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Flatten/Squeeze', 'Unsqueeze']


class Pattern_3(Pattern_1):
    """
    transform
        prev --> Reshape --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape']


class Pattern_4(Pattern_1):
    """
    transform
        prev --> Reshape --> Expand --> Expand --> Reshape --> next

    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Expand', 'Expand', 'Reshape']


class Pattern_5(Pattern_1):
    """
    transform
        prev --> Reshape --> Expand --> Reshape --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Expand', 'Reshape']


class Pattern_6(Pattern_1):
    """
    transform
        prev --> Reshape --> Reshape --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Reshape']


class Pattern_7(Pattern_1):
    """
    transform
        prev --> Reshape --> Reshape --> Reshape --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Reshape', 'Reshape']


class Pattern_8(Pattern_1):
    """
    transform
        prev --> Expand --> next
    to
        prev --> (   ) --> next
    if prev.output[0].shape == next.input[0].shape
    """

    pattern_to_match = ['Expand']
