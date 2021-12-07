import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class ConvertConv1dToConv2d(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Reshape --> Conv --> Reshape --> next
    to
        prev --> Reshape --> Conv --> Reshape --> next

    if Conv.input[0].ndim == 3, i.e., if Conv1d
    """

    pattern_to_match = ['Reshape', 'Conv', 'Reshape']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node, mid_node, base_node = matched_nodes

        # Inserts an axis of length one at the position of the H axis.
        # In other words, a tensor of shape (N, C, D) will be reshaped
        # into (N, C, 1, D). We chose the H axis as an insertion point
        # rather than the W axis, i.e. (N, C, D, 1), because the current
        # implemention of our compiler parallelizes computation with
        # tensors with a large W axis better than those with a large H
        # axis. Refer to
        # <https://github.com/furiosa-ai/dss/pull/183#issuecomment-861137438>.
        new_mid_input_shape = self.get_value_info_shape(mid_node.input[0])
        new_mid_input_shape.insert(-1, 1)
        new_top_reshape_shape = self.get_initializer_array(top_node.input[1]).tolist()
        new_top_reshape_shape.insert(-1, 1)
        new_mid_output_shape = self.get_value_info_shape(mid_node.output[0])
        new_mid_output_shape.insert(-1, 1)
        new_mid_weight_shape = self.get_value_info_shape(mid_node.input[1])
        new_mid_weight_shape.insert(-1, 1)
        self.transform_to_convert(
            matched_nodes,
            nodes_to_add=[
                self.make_node(
                    'Reshape',
                    [top_node.input[0], top_node.input[1] + '_converted'],
                    [top_node.output[0]],
                    top_node.name,
                ),
                self.make_node(
                    'Conv',
                    [
                        mid_node.input[0],
                        mid_node.input[1] + '_converted',
                        mid_node.input[2] if len(mid_node.input) == 3 else None,
                    ],
                    [mid_node.output[0]],
                    mid_node.name,
                    **self.get_attrs(mid_node),
                ),
                base_node,
            ],
            inits_to_add=[
                self.make_initializer_from_array(
                    np.array(new_top_reshape_shape), name=top_node.input[1] + '_converted'
                ),
                self.make_initializer_from_array(
                    self.get_initializer_array(mid_node.input[1]).reshape(new_mid_weight_shape),
                    name=mid_node.input[1] + '_converted',
                ),
                self.initializer_map[mid_node.input[0]] if len(mid_node.input) == 3 else None,
            ],
            vis_to_add=[
                self.make_tensor_value_info(
                    mid_node.input[0], onnx.TensorProto.FLOAT, new_mid_input_shape
                ),
                self.make_tensor_value_info(
                    mid_node.output[0], onnx.TensorProto.FLOAT, new_mid_output_shape
                ),
            ],
        )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        _, mid_node, _ = nodes_to_check
        if len(self.get_value_info_shape(mid_node.input[0])) == 3:
            return True
        return False

    def get_attrs(self, mid_node):
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in mid_node.attribute}
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        dilations = attrs.get('dilations', [1])
        group = attrs.get('group', 1)
        kernel_shape = attrs['kernel_shape']
        strides = attrs.get('strides', [1])

        new_attrs = {
            'auto_pad': auto_pad,
            'dilations': [1, *dilations],
            'group': group,
            'kernel_shape': [1, *kernel_shape],
            'strides': [1, strides[0]],
        }

        if auto_pad == 'NOTSET':
            pads = attrs.get('pads', [0, 0])
            new_attrs.update(
                {
                    'pads': [0, pads[0], 0, pads[1]],
                }
            )

        return new_attrs
