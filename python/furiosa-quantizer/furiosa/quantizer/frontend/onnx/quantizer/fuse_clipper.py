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
            Pattern_5,
            Pattern_6,
        ]:
            model = transformer(model).transform()

        return model


class ClipperFusion(ONNXTransformer):
    """
    This class contains methods commonly used in FuseClipper patterns
    """

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)

        if not matched_nodes:
            return base_node.input

        top_node = matched_nodes[0]
        self.transform_to_fuse(matched_nodes, nodes_to_add=self.make_nodes_to_add(matched_nodes))

        return top_node.input

    def make_nodes_to_add(self, matched_nodes):
        *nodes, node, clip = matched_nodes
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        fused_node = self.make_node(
            node.op_type,
            node.input,
            clip.output,
            node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
        )
        nodes.append(fused_node)
        return nodes


class Pattern_1(ClipperFusion):
    """
    transform
        prev --> Conv --> Relu --> next
    to
        prev --> Conv --> next
    """

    pattern_to_match = ['Conv', 'Relu']


class Pattern_2(ClipperFusion):
    """
    transform
        prev --> Conv --> Clip --> next
    to
        prev --> Conv --> next
    """

    pattern_to_match = ['Conv', 'Clip']


class Pattern_3(ClipperFusion):
    """
    transform
        prev --> Add --> Relu --> next
    to
        prev --> Add --> next
    """

    pattern_to_match = ['Add', 'Relu']


class Pattern_4(ClipperFusion):
    """
    transform
        prev --> Add --> Clip --> next
    to
        prev --> Add --> next
    """

    pattern_to_match = ['Add', 'Clip']


class Pattern_5(ClipperFusion):
    """
    transform
        prev --> Conv --> Squeeze --> Relu --> next
    to
        prev --> Conv --> Squeeze --> next
    """

    pattern_to_match = ['Conv', 'Squeeze', 'Relu']


class Pattern_6(ClipperFusion):
    """
    transform
        prev --> Conv --> Squeeze --> Clip --> next
    to
        prev --> Conv --> Squeeze --> next
    """

    pattern_to_match = ['Conv', 'Squeeze', 'Clip']
