import logging
from typing import Callable, Iterable, List, Tuple

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer, utils
from furiosa.quantizer.interfaces.transformer import Transformer

logger = logging.getLogger(__name__)


class FuseBatchNorm(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [Pattern_1, Pattern_2, Pattern_3, Pattern_4]:
            model = transformer(model).transform()

        return model


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

        conv, batch_norm = matched_nodes
        multiplier, shifter = _get_multiplier_and_shifter(
            *_get_bn_params(batch_norm, self.get_initializer_array)
        )

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=_make_bn_fused_node(conv, batch_norm.output[0]),
            inits_to_add=_make_bn_fused_init(conv, multiplier, shifter, self.get_initializer_array),
            vis_to_add=[],
        )

        return conv.input


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

        conv_trans, batch_norm = matched_nodes
        multiplier, shifter = _get_multiplier_and_shifter(
            *_get_bn_params(batch_norm, self.get_initializer_array)
        )

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=_make_bn_fused_node(conv_trans, batch_norm.output[0]),
            inits_to_add=_make_bn_fused_init(
                conv_trans, multiplier, shifter, self.get_initializer_array
            ),
            vis_to_add=[],
        )

        return conv_trans.input


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
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        conv, mul, add = matched_nodes
        multiplier = self.get_initializer_array(self.get_init_node_input(mul)).flatten()
        shifter = self.get_initializer_array(self.get_init_node_input(add)).flatten()

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=_make_bn_fused_node(conv, add.output[0]),
            inits_to_add=_make_bn_fused_init(conv, multiplier, shifter, self.get_initializer_array),
            vis_to_add=[],
        )

        return conv.input

    def pattern_condition_checker(self, nodes_to_check: List[onnx.NodeProto]) -> bool:
        _, mul, add = nodes_to_check

        # This checks if a node has a initializer, \
        # assuming a node has exactly one initializer if it has one. \
        # That is, there is no node with two initialzier.
        return self.get_init_node_input(mul) and self.get_init_node_input(add)


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

        batch_norm = matched_nodes[0]
        if utils.is_op_type(self.find_prev_node(batch_norm.input[0]), ['Conv']):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=_make_new_node_pattern_4(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
            vis_to_add=self.make_new_vi(matched_nodes),
        )

        return batch_norm.input

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        (batch_norm,) = matched_nodes
        bn_params = _get_bn_params(batch_norm, self.get_initializer_array)
        multiplier, shifter = _get_multiplier_and_shifter(*bn_params)
        num_features = self.get_value_info_shape(batch_norm.output[0])[0]
        return [
            onnx.numpy_helper.from_array(
                multiplier.reshape(num_features, -1, 1, 1),
                name=batch_norm.output[0] + '_bn_multiplier',
            ),
            onnx.numpy_helper.from_array(
                shifter.reshape(num_features, -1, 1, 1), name=batch_norm.output[0] + '_bn_shifter'
            ),
        ]

    def make_new_vi(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        (batch_norm,) = matched_nodes
        return [
            onnx.helper.make_tensor_value_info(
                batch_norm.output[0] + '_bn_multiplied',
                onnx.TensorProto.FLOAT,
                shape=self.get_value_info_shape(batch_norm.output[0]),
            )
        ]


def _get_bn_params(
    node: onnx.NodeProto, get_init_arr_func: Callable[[str], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scale = get_init_arr_func(node.input[1])
    if all(v == 0.0 for v in scale):
        logger.warning('BatchNormalization.scale is a zero tensor: %s', node.input[1])

    B = get_init_arr_func(node.input[2])
    mean = get_init_arr_func(node.input[3])
    var = get_init_arr_func(node.input[4])

    eps = utils.get_attribute(node.attribute, "epsilon", 1e-05)

    return scale, B, mean, var, eps


def _fuse_bn_weight(weight: np.ndarray, multiplier: np.ndarray, axis: int = 0) -> np.ndarray:
    idx = [1, 1, 1]
    idx.insert(axis, -1)
    return weight * multiplier.reshape(idx)


def _fuse_bn_bias(bias: np.ndarray, multiplier: np.ndarray, shifter: np.ndarray):
    return bias * multiplier + shifter


def _get_multiplier_and_shifter(
    scale: np.ndarray, B: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    reciprocal = 1 / np.sqrt(var + eps)
    multiplier = scale * reciprocal
    shifter = -mean * scale * reciprocal + B
    return multiplier, shifter


def _make_bn_fused_node(node: onnx.NodeProto, output_tensor: str) -> List[onnx.NodeProto]:
    assert node.op_type in ['Conv', 'ConvTranspose'], repr(node)
    input_names = [node.input[0], node.input[1] + '_bn_fused']
    if len(node.input) == 3:
        input_names.append(node.input[2] + '_bn_fused')
    else:
        input_names.append(node.output[0] + '_bias_bn_fused')

    return [
        onnx.helper.make_node(
            node.op_type,
            input_names,
            [output_tensor],
            node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute},
        )
    ]


def _make_bn_fused_init(
    node: onnx.NodeProto,
    multiplier: np.ndarray,
    shifter: np.ndarray,
    get_init_arr_func: Callable[[str], np.ndarray],
) -> List[onnx.TensorProto]:
    assert node.op_type in ['Conv', 'ConvTranspose'], repr(node)
    weight = get_init_arr_func(node.input[1])
    fused_weight = _fuse_bn_weight(weight, multiplier, axis=0 if node.op_type == 'Conv' else 1)
    fused_weight_name = node.input[1] + '_bn_fused'

    if len(node.input) == 3:
        bias = get_init_arr_func(node.input[2])
        fused_bias_name = node.input[2] + '_bn_fused'
    else:
        bias = np.zeros_like(shifter)
        fused_bias_name = node.output[0] + '_bias_bn_fused'
    fused_bias = _fuse_bn_bias(bias, multiplier, shifter)

    return [
        onnx.numpy_helper.from_array(fused_weight, name=fused_weight_name),
        onnx.numpy_helper.from_array(fused_bias, name=fused_bias_name),
    ]


def _make_new_node_pattern_4(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
    (batch_norm,) = matched_nodes
    return [
        onnx.helper.make_node(
            'Mul',
            [batch_norm.input[0], batch_norm.output[0] + '_bn_multiplier'],
            [batch_norm.output[0] + '_bn_multiplied'],
            batch_norm.name,
        ),
        onnx.helper.make_node(
            'Add',
            [batch_norm.output[0] + '_bn_multiplied', batch_norm.output[0] + '_bn_shifter'],
            [batch_norm.output[0]],
            batch_norm.name,
        ),
    ]
