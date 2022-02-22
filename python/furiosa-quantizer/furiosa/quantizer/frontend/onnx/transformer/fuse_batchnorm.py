import logging
from typing import List, Tuple

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer

logger = logging.getLogger(__name__)


class FuseBatchNorm(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [Pattern_1, Pattern_2, Pattern_3, Pattern_4]:
            model = transformer(model).transform()

        return model


def _get_bn_params(
    node: onnx.NodeProto, get_init_arr_func: classmethod
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scale = get_init_arr_func(node.input[1])
    if all(v == 0.0 for v in scale):
        logger.warning(f'BatchNormalization.scale is a zero tensor: {node.input[1]}')

    B = get_init_arr_func(node.input[2])
    mean = get_init_arr_func(node.input[3])
    var = get_init_arr_func(node.input[4])

    eps = next(
        (
            onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
            if attr.name == "epsilon"
        ),
        1e-05,
    )

    return scale, B, mean, var, eps


def _fuse_bn_params(
    weight: np.ndarray, multiplier: np.ndarray, shifter: np.ndarray, axis: int = 0
) -> np.ndarray:
    if weight.ndim == 4:
        idx = [1, 1, 1]
        idx.insert(axis, -1)
        fused_weight = weight * multiplier.reshape(idx)
        return fused_weight
    elif weight.ndim == 1:
        fused_bias = weight * multiplier + shifter
        return fused_bias
    else:
        raise Exception('Unknown weight ndim: %s' % weight.dim)


def _get_multiplier_and_shifter(
    scale: np.ndarray, B: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    reciprocal = np.sqrt(var + eps)
    multiplier = scale / reciprocal
    shifter = -mean * scale / reciprocal + B

    return multiplier, shifter


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Conv --> BatchNormalization --> next
    to
        prev --> Conv --> next
    """

    pattern_to_match = ['Conv', 'BatchNormalization']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=[],
        )

        return top_node.input

    def make_new_node(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.NodeProto]:
        top_node, base_node = matched_nodes

        input_names = [
            node_input if node_input not in self.initializer_map else node_input + '_bn_fused'
            for node_input in top_node.input
        ]

        return [self.make_node('Conv', input_names, [base_node.output[0]], top_node.name)]

    def make_new_init(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.TensorProto]:
        top_node, base_node = matched_nodes
        bn_params = _get_bn_params(base_node, self.get_initializer_array)
        multiplier, shifter = _get_multiplier_and_shifter(*bn_params)

        inits_to_add = []
        for node_input in top_node.input:
            if node_input not in self.initializer_map:
                continue
            weight = self.get_initializer_array(node_input)
            fused_weight = _fuse_bn_params(weight, multiplier, shifter)
            inits_to_add.append(
                self.make_initializer_from_array(fused_weight, node_input + '_bn_fused')
            )

        return inits_to_add


class Pattern_2(ONNXTransformer):
    """
    transform
        prev --> ConvTranspose --> BatchNormalization --> next
    to
        prev --> ConvTranspose --> next
    """

    pattern_to_match = ['ConvTranspose', 'BatchNormalization']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=[],
        )

        return top_node.input

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes

        input_names = [
            node_input if node_input not in self.initializer_map else node_input + '_bn_fused'
            for node_input in top_node.input
        ]

        return [
            self.make_node(
                'ConvTranspose',
                input_names,
                [base_node.output[0]],
                top_node.name,
                **{attr.name: onnx.helper.get_attribute_value(attr) for attr in top_node.attribute},
            )
        ]

    def make_new_init(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.TensorProto]:
        top_node, base_node = matched_nodes
        bn_params = _get_bn_params(base_node, self.get_initializer_array)
        multiplier, shifter = _get_multiplier_and_shifter(*bn_params)

        inits_to_add = []
        for node_input in top_node.input:
            if node_input not in self.initializer_map:
                continue
            weight = self.get_initializer_array(node_input)
            fused_weight = _fuse_bn_params(weight, multiplier, shifter, axis=1)
            inits_to_add.append(
                self.make_initializer_from_array(fused_weight, node_input + '_bn_fused')
            )

        return inits_to_add


class Pattern_3(ONNXTransformer):
    """
    transform
        prev --> Conv --> Mul --> Add --> next
    to
        prev --> Conv --> next

    if 1. Mul has only one initializer
       2. Add has only one initializer
    """

    pattern_to_match = ['Conv', 'Mul', 'Add']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        inputs = base_node.input
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node = matched_nodes[0]
        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=[],
        )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check: List[onnx.NodeProto]) -> bool:
        _, mul_node, add_node = nodes_to_check

        # This checks if a node has a initializer, \
        # assuming a node has exactly one initializer if it has one. \
        # That is, there is no node with two initialzier.
        return self.get_init_node_input(mul_node) and self.get_init_node_input(add_node)

    def make_new_node(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.NodeProto]:
        top_node, _, bottom_node = matched_nodes

        input_names = []
        input_names.append(top_node.input[0])
        input_names.append(top_node.input[1] + '_bn_fused')
        if len(top_node.input) == 3:
            input_names.append(top_node.input[2] + '_bn_fused')
        else:
            input_names.append(top_node.output[0] + '_bias_bn_fused')

        return [
            self.make_node(
                'Conv',
                input_names,
                [bottom_node.output[0]],
                top_node.name,
                **{attr.name: onnx.helper.get_attribute_value(attr) for attr in top_node.attribute},
            )
        ]

    def make_new_init(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.TensorProto]:
        top_node, middle_node, bottom_node = matched_nodes
        multiplier, shifter = self.get_multiplier_and_shifter(middle_node, bottom_node)

        weight = self.get_initializer_array(top_node.input[1])
        fused_weight = _fuse_bn_params(weight, multiplier, shifter)
        fused_weight_name = top_node.input[1] + '_bn_fused'

        if len(top_node.input) == 3:
            bias = self.get_initializer_array(top_node.input[2])
            fused_bias_name = top_node.input[2] + '_bn_fused'
        else:
            bias = np.zeros(weight.shape[0]).astype(np.float32)
            fused_bias_name = top_node.output[0] + '_bias_bn_fused'
        fused_bias = _fuse_bn_params(bias, multiplier, shifter)

        return [
            self.make_initializer_from_array(fused_weight, name=fused_weight_name),
            self.make_initializer_from_array(fused_bias, name=fused_bias_name),
        ]

    def get_multiplier_and_shifter(self, mul_node, add_node):
        multiplier = self.get_initializer_array(self.get_init_node_input(mul_node))
        shifter = self.get_initializer_array(self.get_init_node_input(add_node))
        return multiplier.flatten(), shifter.flatten()


class Pattern_4(ONNXTransformer):
    """
    transform
        prev --> BatchNormalization --> next
    to
        prev --> Mul --> Add --> next

    if prev.op_type != Conv
    """

    pattern_to_match = ['BatchNormalization']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=self.make_new_vi(matched_nodes),
        )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check: List[onnx.NodeProto]) -> bool:
        node = nodes_to_check[0]

        if self.is_op_type(node.op_type, ['Conv']):
            return False
        return True

    def make_new_node(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.NodeProto]:
        node = matched_nodes[0]
        return [
            self.make_node(
                'Mul',
                [node.input[0], node.output[0] + '_bn_multiplier'],
                [node.output[0] + '_bn_multiplied'],
                node.name,
            ),
            self.make_node(
                'Add',
                [node.output[0] + '_bn_multiplied', node.output[0] + '_bn_shifter'],
                [node.output[0]],
                node.name,
            ),
        ]

    def make_new_init(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.TensorProto]:
        node = matched_nodes[0]
        bn_params = _get_bn_params(node, self.get_initializer_array)
        multiplier, shifter = _get_multiplier_and_shifter(*bn_params)
        num_features = self.get_value_info_shape(node.output[0])[0]
        return [
            self.make_initializer_from_array(
                multiplier.reshape(num_features, -1, 1, 1), name=node.output[0] + '_bn_multiplier'
            ),
            self.make_initializer_from_array(
                shifter.reshape(num_features, -1, 1, 1), name=node.output[0] + '_bn_shifter'
            ),
        ]

    def make_new_vi(self, matched_nodes: List[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        node = matched_nodes[0]
        return [
            self.make_tensor_value_info(
                node.output[0] + '_bn_multiplied',
                onnx.TensorProto.FLOAT,
                shape=self.get_value_info_shape(node.output[0]),
            )
        ]
