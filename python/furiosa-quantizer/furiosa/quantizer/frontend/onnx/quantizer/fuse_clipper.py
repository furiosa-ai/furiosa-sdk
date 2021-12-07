import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseClipper(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
            Pattern_3,
            Pattern_4,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Conv --> Relu --> next
    to
        prev --> Conv --> next
    """

    pattern_to_match = ['Conv', 'Relu']

    def pattern_matching(self, base_node):
        inputs = base_node.input
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        top_node = matched_nodes[0]
        self.transform_to_fuse(matched_nodes, nodes_to_add=[self.make_new_node(matched_nodes)])

        return top_node.input

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes
        return self.make_node(
            'Conv',
            top_node.input,
            [base_node.output[0]],
            top_node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in top_node.attribute},
        )


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Conv --> Clip --> next
    to
        prev --> Conv --> next
    """

    pattern_to_match = ['Conv', 'Clip']


class Pattern_3(Pattern_1):
    """
    transform
        prev --> Add --> Relu --> next
    to
        prev --> Add --> next
    """

    pattern_to_match = ['Add', 'Relu']

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes
        return self.make_node(
            'Add',
            top_node.input,
            [base_node.output[0]],
            top_node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in top_node.attribute},
        )


class Pattern_4(Pattern_3):
    """
    transform
        prev --> Add --> Clip --> next
    to
        prev --> Add --> next
    """

    pattern_to_match = ['Add', 'Clip']
