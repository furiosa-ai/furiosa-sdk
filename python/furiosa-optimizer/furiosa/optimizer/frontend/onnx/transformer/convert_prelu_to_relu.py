from typing import Iterable, List

import numpy as np
import onnx

from furiosa.optimizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.optimizer.interfaces.transformer import Transformer


class ConvertPReluToRelu(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:  # pylint: disable=no-member
        for transformer in [Pattern_1, Pattern_2]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        PRelu(x) = slope * x if x < 0, x if x >=0
    into
        (1 - slope) * Relu(x) + slope * x
    if
        1. PRelu's input[1] is an initializer
    """

    pattern_to_match = ['PRelu']

    def pattern_matching(
        self, base_node: onnx.NodeProto  # pylint: disable=no-member
    ) -> Iterable[str]:
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

        return base_node.input

    def pattern_condition_checker(
        self, nodes_to_check: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> bool:
        (prelu,) = nodes_to_check

        return prelu.input[1] in self.initializer_map

    def make_new_node(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes

        relu = onnx.helper.make_node(
            'Relu',
            inputs=[prelu.input[0]],
            outputs=[prelu.output[0] + '_relu_out'],
            name=prelu.output[0] + '_0',
        )

        mul_0 = onnx.helper.make_node(
            'Mul',
            inputs=[prelu.output[0] + '_relu_out', prelu.output[0] + '_slope_fused'],
            outputs=[prelu.output[0] + '_mul_0_out'],
            name=prelu.output[0] + '_1',
        )

        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[prelu.input[0], prelu.input[1]],
            outputs=[prelu.output[0] + '_mul_1_out'],
            name=prelu.output[0] + '_2',
        )

        add = onnx.helper.make_node(
            'Add',
            inputs=[prelu.output[0] + '_mul_0_out', prelu.output[0] + '_mul_1_out'],
            outputs=prelu.output,
            name=prelu.output[0] + '_3',
        )

        return [relu, mul_0, mul_1, add]

    def make_new_init(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes

        slope = self.get_initializer_array(prelu.input[1])
        one_minus_slope = np.ones_like(slope) - slope
        new_init = onnx.numpy_helper.from_array(one_minus_slope, prelu.output[0] + '_slope_fused')

        return [new_init]

    def make_new_vi(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes
        input_shape = self.get_value_info_shape(prelu.input[0])
        input_dtype = self.get_value_info_dtype(prelu.input[0])

        return [
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_relu_out',
                input_dtype,
                input_shape,
            ),
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_mul_0_out',
                input_dtype,
                input_shape,
            ),
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_mul_1_out',
                input_dtype,
                input_shape,
            ),
        ]


class Pattern_2(ONNXTransformer):
    """
    transform
        PRelu(x) = slope * x if x < 0, x if x >=0
    into
        (1 - slope) * Relu(x) + slope * x
    if
        1. PRelu's input[1] is not an initializer
    """

    pattern_to_match = ['PRelu']

    def pattern_matching(
        self, base_node: onnx.NodeProto  # pylint: disable=no-member
    ) -> Iterable[str]:
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

        return base_node.input

    def pattern_condition_checker(
        self, nodes_to_check: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> bool:
        (prelu,) = nodes_to_check

        return prelu.input[1] not in self.initializer_map

    def make_new_node(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes

        relu = onnx.helper.make_node(
            'Relu',
            inputs=[prelu.input[0]],
            outputs=[prelu.output[0] + '_relu_out'],
            name=prelu.output[0] + '_0',
        )

        sub = onnx.helper.make_node(
            'Sub',
            inputs=[prelu.output[0] + '_ones_like_slope', prelu.input[1]],
            outputs=[prelu.output[0] + '_slope_fused'],
            name=prelu.output[0] + '_1',
        )

        mul_0 = onnx.helper.make_node(
            'Mul',
            inputs=[prelu.output[0] + '_relu_out', prelu.output[0] + '_slope_fused'],
            outputs=[prelu.output[0] + '_mul_0_out'],
            name=prelu.output[0] + '_2',
        )

        mul_1 = onnx.helper.make_node(
            'Mul',
            inputs=[prelu.input[0], prelu.input[1]],
            outputs=[prelu.output[0] + '_mul_1_out'],
            name=prelu.output[0] + '_3',
        )

        add = onnx.helper.make_node(
            'Add',
            inputs=[prelu.output[0] + '_mul_0_out', prelu.output[0] + '_mul_1_out'],
            outputs=prelu.output,
            name=prelu.output[0] + '_4',
        )

        return [relu, sub, mul_0, mul_1, add]

    def make_new_init(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes

        slope_shape = self.get_value_info_shape(prelu.input[1])
        slope_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[self.get_value_info_dtype(prelu.input[1])]

        ones_like_slope = np.ones(slope_shape).astype(slope_dtype)
        new_init = onnx.numpy_helper.from_array(
            ones_like_slope, prelu.output[0] + '_ones_like_slope'
        )

        return [new_init]

    def make_new_vi(
        self, matched_nodes: Iterable[onnx.NodeProto]  # pylint: disable=no-member
    ) -> List[onnx.NodeProto]:  # pylint: disable=no-member
        (prelu,) = matched_nodes
        input_shape = self.get_value_info_shape(prelu.input[0])
        slope_shape = self.get_value_info_shape(prelu.input[1])
        input_dtype = self.get_value_info_dtype(prelu.input[0])

        return [
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_relu_out',
                input_dtype,
                input_shape,
            ),
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_slope_fused',
                input_dtype,
                slope_shape,
            ),
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_mul_0_out',
                input_dtype,
                input_shape,
            ),
            onnx.helper.make_tensor_value_info(
                prelu.output[0] + '_mul_1_out',
                input_dtype,
                input_shape,
            ),
        ]
