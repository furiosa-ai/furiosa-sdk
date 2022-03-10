import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FusePad(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [Pattern_1, Pattern_2]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Pad --> MaxPool --> next
    to
        prev --> MaxPool --> next

    if 1. Pad.mode == 'constant'
       2. Pad.constant_value == -inf
       3. padded on spatial dimension
       4. fused_pads[i] < kernel_shape[i] and fused_pads[i + kernel_rank] < kernel_shape[i] for all i
    """

    pattern_to_match = ['Pad', 'MaxPool']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        self.transform_to_fuse(matched_nodes, nodes_to_add=[self.make_new_node(matched_nodes)])

        return matched_nodes[0].input

    def pattern_condition_checker(self, nodes_to_check):
        top_node, base_node = nodes_to_check

        if not self.check_condition_1(top_node.attribute):
            return False

        if not self.check_condition_2(top_node):
            return False

        if not self.check_condition_3(top_node.input[1]):
            return False

        if not self.check_condition_6(self.get_attrs(base_node), top_node.input[1]):
            return False

        return True

    def check_condition_1(self, node_attr):
        if self.get_pad_mode(node_attr) == 'constant':
            return True
        return False

    def check_condition_2(self, node):
        try:
            const_input = node.input[2]
            constant_value = self.get_initializer_array(const_input)
        except IndexError:
            constant_value = 0.0

        if constant_value == float('-inf'):
            return True
        return False

    def check_condition_3(self, pads_input):
        pads = self.get_initializer_array(pads_input)
        rank = len(pads) // 2
        pads_on_nc_dim = [*pads[:2], *pads[rank : rank + 2]]

        if all(pad == 0 for pad in pads_on_nc_dim):
            return True
        return False

    def check_condition_6(self, node_attrs, pad_input):
        attrs = self.update_attrs(node_attrs, pad_input)
        kernel_shape = attrs['kernel_shape']
        kernel_rank = len(kernel_shape)
        fused_pads = attrs['pads']
        fused_pad_shape = [
            sum([fused_pads[dim], fused_pads[dim + kernel_rank]]) for dim in range(kernel_rank)
        ]

        assert len(kernel_shape) == len(fused_pad_shape)
        if all(
            fused_pads[dim] < k and fused_pads[dim + kernel_rank] < k
            for dim, k in enumerate(kernel_shape)
        ):
            return True
        return False

    def get_pad_mode(self, node_attr):
        return next(
            (onnx.helper.get_attribute_value(attr) for attr in node_attr if attr.name == "mode"),
            b"constant",
        ).decode("utf-8")

    def update_attrs(self, attrs, pad_input):
        pads = [sum(x) for x in zip(attrs['pads'], self.make_maxpool_pad(pad_input))]
        attrs['pads'] = pads

        return attrs

    def get_attrs(self, node):
        rank = len(self.get_value_info_shape(node.input[0]))
        nspatial_dim = rank - 2

        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        ceil_mode = attrs.get('ceil_mode', 0)
        dilations = attrs.get('dilations', [1] * nspatial_dim)
        kernel_shape = attrs['kernel_shape']
        strides = attrs.get('strides', [1] * nspatial_dim)
        pads = attrs.get('pads', [0] * nspatial_dim * 2)

        return {
            'ceil_mode': ceil_mode,
            'dilations': dilations,
            'kernel_shape': kernel_shape,
            'pads': pads,
            'strides': strides,
        }

    def make_maxpool_pad(self, pad_input):
        pads = self.get_initializer_array(pad_input)
        rank = len(pads) // 2

        new_pads = []
        for pad in pads:
            if pad == -1:
                new_pads.append(0)
            else:
                new_pads.append(pad)
        pads = new_pads

        return [*pads[2:rank], *pads[rank + 2 : 2 * rank]]

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes
        attrs = self.update_attrs(self.get_attrs(base_node), top_node.input[1])

        return self.make_node(
            'MaxPool', [top_node.input[0]], [base_node.output[0]], name=top_node.name, **attrs
        )


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Pad --> AveragePool --> next
    to
        prev --> AveragePool --> next

    if 1. Pad.mode == 'constant'
       2. Pad.constant_value == 0.0
       3. padded on spatial dimension
       4. AveragePool.count_include_pad == 1 or all AveragePool.pads == 0
       5. AveragePool.ceil_mode == 0
       6. fused_pads[i] < kernel_shape[i] and fused_pads[i + kernel_rank] < kernel_shape[i] for all i
    """

    pattern_to_match = ['Pad', 'AveragePool']

    def pattern_condition_checker(self, nodes_to_check):
        top_node, base_node = nodes_to_check
        return (
            self.check_condition_1(top_node.attribute)
            and self.check_condition_2(top_node)
            and self.check_condition_3(top_node.input[1])
            and self.check_condition_4(base_node)
            and self.check_condition_5(base_node)
            and self.check_condition_6(self.get_attrs(base_node), top_node.input[1])
        )

    def get_attrs(self, node):
        rank = len(self.get_value_info_shape(node.input[0]))
        nspatial_dim = rank - 2

        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        ceil_mode = attrs.get('ceil_mode', 0)
        count_include_pad = attrs.get('count_include_pad', 0)
        kernel_shape = attrs['kernel_shape']
        strides = attrs.get('strides', [1] * nspatial_dim)
        pads = attrs.get('pads', [0] * nspatial_dim * 2)

        return {
            'ceil_mode': ceil_mode,
            'count_include_pad': count_include_pad,
            'kernel_shape': kernel_shape,
            'pads': pads,
            'strides': strides,
        }

    def update_attrs(self, attrs, pad_input):
        pads = [sum(x) for x in zip(attrs['pads'], self.make_maxpool_pad(pad_input))]
        attrs['pads'] = pads
        attrs['count_include_pad'] = 1

        return attrs

    def check_condition_2(self, node):
        try:
            const_input = node.input[2]
            constant_value = self.get_initializer_array(const_input)
        except IndexError:
            constant_value = 0.0

        if constant_value == 0.0:
            return True
        return False

    def check_condition_4(self, node):
        attrs = self.get_attrs(node)
        count_include_pad = attrs['count_include_pad']
        pads = attrs['pads']
        if count_include_pad == 1 or all(pad == 0 for pad in pads):
            return True
        return False

    def check_condition_5(self, node):
        attrs = self.get_attrs(node)
        ceil_mode = attrs['ceil_mode']
        if ceil_mode == 0:
            return True
        return False

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes
        attrs = self.update_attrs(self.get_attrs(base_node), top_node.input[1])

        return self.make_node(
            'AveragePool', [top_node.input[0]], [base_node.output[0]], name=top_node.name, **attrs
        )
