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
    postfix = '_reshape_fused'

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
            nodes_to_add=[self.make_new_node(matched_nodes)],
            inits_to_add=[self.make_new_init(matched_nodes)],
            vis_to_add=[self.make_new_vi(matched_nodes)]
            if self.make_new_vi(matched_nodes)
            else None,
        )
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        top_node = nodes_to_check[0]
        base_node = nodes_to_check[-1]
        if self.is_same_shape(top_node.input[0], base_node.output[0]):
            return False
        return True

    def make_new_node(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]
        return self.make_node(
            'Reshape',
            [top_node.input[0], top_node.input[1] + self.postfix],
            [base_node.output[0]],
            name=top_node.name,
        )

    def make_new_init(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]
        return self.make_int64_initializer(top_node.input[1] + self.postfix, base_node.output[0])

    def make_new_vi(self, matched_nodes):
        return None


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Reshape --> Reshape --> Reshape --> next
    to
        prev --> Reshape --> next

    if prev.output[0].shape != next.input[0].shape
    """

    pattern_to_match = ['Reshape', 'Reshape', 'Reshape']


class Pattern_3(Pattern_1):
    """
    transform
        prev --> Flatten/Squeeze --> Unsqueeze --> next
    to
        prev --> Reshape --> next
    if prev.output[0].shape != next.input[0].shape
    """

    pattern_to_match = ['Flatten/Squeeze', 'Unsqueeze']

    def make_new_node(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]
        return self.make_node(
            'Reshape',
            [top_node.input[0], top_node.input[0] + self.postfix],
            [base_node.output[0]],
            top_node.name,
        )

    def make_new_init(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]
        return self.make_int64_initializer(top_node.input[0] + self.postfix, base_node.output[0])

    def make_new_vi(self, matched_nodes):
        top_node = matched_nodes[0]
        return self.copy_value_info(top_node.input[0])
