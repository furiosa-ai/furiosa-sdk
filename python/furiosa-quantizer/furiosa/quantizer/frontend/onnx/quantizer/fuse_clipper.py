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
            transformer = transformer(model)
            # FuseClipper().transform() is called on QDQ graph.
            # QDQ graph will apply QuantizeLinear - DequantizeLinear to bias of Conv operators.
            # bias of QLinearConv operator should be i32, so QDQ graph will have i32 quantization information for bias.
            # However, QuantizeLinear operator of onnx are only defined on i8 quantization parmeter, so temporarily, QDQ graph is not valid.
            transformer.check_runnable = False
            model = transformer.transform()

        return model


class ClipperFusion(ONNXTransformer):
    """
    This class contains methods commonly used in FuseClipper patterns
    """

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)

        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        top_node = matched_nodes[0]
        self.transform_to_fuse(matched_nodes, nodes_to_add=self.make_nodes_to_add(matched_nodes))
        self.remove_clip_qdq(matched_nodes[-2])

        return top_node.input

    def pattern_condition_checker(self, matched_nodes):
        clip = matched_nodes[-2]
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        if clip.op_type == "Relu":
            return True
        min_tensor = self.find_prev_node(self.find_prev_node(clip.input[1]).input[0]).input[0]
        max_tensor = self.find_prev_node(self.find_prev_node(clip.input[2]).input[0]).input[0]
        return min_tensor in self.initializer_map and max_tensor in self.initializer_map

    def remove_clip_qdq(self, clip):
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        prev_nodes = []
        for input in clip.input:
            dqlinear = self.traverse_prev_node(input, ["DequantizeLinear"])
            qlinear = self.traverse_prev_node(dqlinear.input[0], ["QuantizeLinear"])
            prev_nodes.extend([dqlinear, qlinear])

        self.pop_multiple_optimizer_map(prev_nodes)

    def make_nodes_to_add(self, matched_nodes):
        *nodes, node, _, _, clip, qlinear = matched_nodes
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        fused_node = self.make_node(
            node.op_type,
            node.input,
            clip.output,
            node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
        )
        nodes.append(fused_node)
        nodes.append(qlinear)
        return nodes


class Pattern_1(ClipperFusion):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> next
    """

    pattern_to_match = ['Conv', 'QuantizeLinear', 'DequantizeLinear', 'Relu', 'QuantizeLinear']


class Pattern_2(ClipperFusion):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = ['Conv', 'QuantizeLinear', 'DequantizeLinear', 'Clip', 'QuantizeLinear']


class Pattern_3(ClipperFusion):
    """
    transform
        prev --> Add --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Add --> QuantizeLinear --> next
    """

    pattern_to_match = ['Add', 'QuantizeLinear', 'DequantizeLinear', 'Relu', 'QuantizeLinear']


class Pattern_4(ClipperFusion):
    """
    transform
        prev --> Add --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Add --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = ['Add', 'QuantizeLinear', 'DequantizeLinear', 'Clip', 'QuantizeLinear']


class Pattern_5(ClipperFusion):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> next
    """

    pattern_to_match = [
        'Conv',
        'QuantizeLinear',
        'DequantizeLinear',
        'Squeeze',
        'QuantizeLinear',
        'DequantizeLinear',
        'Relu',
        'QuantizeLinear',
    ]


class Pattern_6(ClipperFusion):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = [
        'Conv',
        'QuantizeLinear',
        'DequantizeLinear',
        'Squeeze',
        'QuantizeLinear',
        'DequantizeLinear',
        'Clip',
        'QuantizeLinear',
    ]
