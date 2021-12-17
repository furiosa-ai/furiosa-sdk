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
        prev --> Reshape --> Conv(1d) --> Reshape --> next
    to
        prev --> Reshape --> Conv(2d) --> Reshape --> next

    if Conv(1d).input[0].ndim == 3
    """

    pattern_to_match = ['Reshape', 'Conv', 'Reshape']

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        reshape_0, conv1d, reshape_1 = matched_nodes

        # Inserts an axis of length one at the position of the H axis.
        # In other words, a tensor of shape (N, C, D) will be reshaped
        # into (N, C, 1, D). We chose the H axis as an insertion point
        # rather than the W axis, i.e. (N, C, D, 1), because the current
        # implemention of our compiler parallelizes computation with
        # tensors with a large W axis better than those with a large H
        # axis. Refer to
        # <https://github.com/furiosa-ai/dss/pull/183#issuecomment-861137438>.
        conv2d_input_shape = self.get_value_info_shape(conv1d.input[0])
        conv2d_input_shape.insert(-1, 1)
        conv2d_output_shape = self.get_value_info_shape(conv1d.output[0])
        conv2d_output_shape.insert(-1, 1)
        conv2d_weight_shape = self.get_value_info_shape(conv1d.input[1])
        conv2d_weight_shape.insert(-1, 1)
        new_reshape_0 = self.get_initializer_array(reshape_0.input[1]).tolist()
        new_reshape_0.insert(-1, 1)
        self.transform_to_convert(
            matched_nodes,
            nodes_to_add=[
                self.make_node(
                    'Reshape',
                    [reshape_0.input[0], reshape_0.input[1] + '_converted'],
                    [reshape_0.output[0]],
                    reshape_0.name,
                ),
                self.make_node(
                    'Conv',
                    [
                        conv1d.input[0],
                        conv1d.input[1] + '_converted',
                        conv1d.input[2] if len(conv1d.input) == 3 else None,
                    ],
                    [conv1d.output[0]],
                    conv1d.name,
                    **self.get_conv2d_attrs(conv1d),
                ),
                reshape_1,
            ],
            inits_to_add=[
                self.make_initializer_from_array(
                    np.array(new_reshape_0), name=reshape_0.input[1] + '_converted'
                ),
                self.make_initializer_from_array(
                    self.get_initializer_array(conv1d.input[1]).reshape(conv2d_weight_shape),
                    name=conv1d.input[1] + '_converted',
                ),
                self.initializer_map[conv1d.input[0]] if len(conv1d.input) == 3 else None,
            ],
            vis_to_add=[
                self.make_tensor_value_info(
                    conv1d.input[0], onnx.TensorProto.FLOAT, conv2d_input_shape
                ),
                self.make_tensor_value_info(
                    conv1d.output[0], onnx.TensorProto.FLOAT, conv2d_output_shape
                ),
            ],
        )

        return reshape_0.input

    def pattern_condition_checker(self, nodes_to_check):
        _, conv1d, _ = nodes_to_check
        return len(self.get_value_info_shape(conv1d.input[0])) == 3

    def get_conv2d_attrs(self, conv1d):
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv1d.attribute}
        auto_pad = attrs.get('auto_pad', 'NOTSET')
        dilations = attrs.get('dilations', [1])
        group = attrs.get('group', 1)
        kernel_shape = attrs.get('kernel_shape', [self.get_value_info_shape(conv1d.input[1])[-1]])
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
