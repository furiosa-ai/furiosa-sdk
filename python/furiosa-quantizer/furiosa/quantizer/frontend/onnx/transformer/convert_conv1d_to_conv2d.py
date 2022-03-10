from typing import Dict, Iterable, List, Sequence

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
        prev --> Conv(1d) --> next
    to
        prev --> Reshape --> Conv(2d) --> Reshape --> next

    if Conv(1d).input[0].ndim == 3
    """

    pattern_to_match = ['Conv']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=self.make_new_vi(matched_nodes),
        )
        conv1d = matched_nodes[0]
        return conv1d.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        (conv1d,) = nodes_to_check
        return len(self.get_value_info_shape(conv1d.input[0])) == 3

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        (conv1d,) = matched_nodes
        reshape_0 = onnx.helper.make_node(
            'Reshape',
            [conv1d.input[0], conv1d.input[0] + '_reshape'],
            [conv1d.input[0] + '_reshaped'],
            conv1d.name + '_0',
        )
        conv2d = onnx.helper.make_node(
            'Conv',
            [
                reshape_0.output[0],
                conv1d.input[1] + '_expanded',
                conv1d.input[2] if len(conv1d.input) == 3 else "",
            ],
            [conv1d.output[0] + '_expanded'],
            conv1d.name + '_1',
            **_get_conv2d_attrs(conv1d, kernel_size=self.get_value_info_shape(conv1d.input[1])[-1]),
        )
        reshape_1 = onnx.helper.make_node(
            'Reshape',
            [conv2d.output[0], conv1d.output[0] + '_reshape'],
            [conv1d.output[0]],
            conv1d.name + '_2',
        )
        return [reshape_0, conv2d, reshape_1]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        (conv1d,) = matched_nodes
        conv2d_weight_shape = self._get_expanded_shape(conv1d.input[1])
        new_reshape_0 = self._get_expanded_shape(conv1d.input[0])
        new_reshape_1 = self.get_value_info_shape(conv1d.output[0])
        return [
            onnx.numpy_helper.from_array(
                np.array(new_reshape_0), name=conv1d.input[0] + '_reshape'
            ),
            onnx.numpy_helper.from_array(
                self.get_initializer_array(conv1d.input[1]).reshape(conv2d_weight_shape),
                name=conv1d.input[1] + '_expanded',
            ),
            self.initializer_map.get(conv1d.input[2]) if len(conv1d.input) == 3 else None,
            onnx.numpy_helper.from_array(
                np.array(new_reshape_1), name=conv1d.output[0] + '_reshape'
            ),
        ]

    def make_new_vi(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        (conv1d,) = matched_nodes
        conv2d_input_shape = self._get_expanded_shape(conv1d.input[0])
        conv2d_output_shape = self._get_expanded_shape(conv1d.output[0])
        return [
            onnx.helper.make_tensor_value_info(
                conv1d.input[0] + '_reshaped', onnx.TensorProto.FLOAT, conv2d_input_shape
            ),
            onnx.helper.make_tensor_value_info(
                conv1d.output[0] + '_expanded', onnx.TensorProto.FLOAT, conv2d_output_shape
            ),
        ]

    def _get_expanded_shape(self, tensor: str) -> Sequence[int]:
        # Inserts an axis of length one at the position of the H axis.
        # In other words, a tensor of shape (N, C, D) will be reshaped
        # into (N, C, 1, D). We chose the H axis as an insertion point
        # rather than the W axis, i.e. (N, C, D, 1), because the current
        # implemention of our compiler parallelizes computation with
        # tensors with a large W axis better than those with a large H
        # axis. Refer to
        # <https://github.com/furiosa-ai/dss/pull/183#issuecomment-861137438>.
        shape = self.get_value_info_shape(tensor)
        shape.insert(-1, 1)
        return shape


def _get_conv2d_attrs(conv1d: onnx.NodeProto, kernel_size: Sequence[int]) -> Dict:
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv1d.attribute}
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    dilations = attrs.get('dilations', [1])
    group = attrs.get('group', 1)
    kernel_shape = attrs.get('kernel_shape', [kernel_size])
    strides = attrs.get('strides', [1])

    new_attrs = {
        'auto_pad': auto_pad,
        'dilations': [1, *dilations],
        'group': group,
        'kernel_shape': [1, *kernel_shape],
        'strides': [1, *strides],
    }

    if auto_pad == 'NOTSET':
        pads = attrs.get('pads', [0, 0])
        new_attrs.update(
            {
                'pads': [0, pads[0], 0, pads[1]],
            }
        )

    return new_attrs
