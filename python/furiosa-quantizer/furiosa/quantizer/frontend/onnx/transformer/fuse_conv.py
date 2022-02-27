from typing import AbstractSet, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


# TODO: consider (init, data) input case for MatMul
# related issue: https://github.com/furiosa-ai/furiosa-sdk-private/issues/243
class FuseConv(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
            Pattern_3,
        ]:
            model = transformer(model).transform()

        return model


def _is_np_broadcastable(array: np.ndarray, to_shape: Sequence[int]) -> bool:
    try:
        np.broadcast_to(array, to_shape)
        return True
    except ValueError:
        return False


def _matmul_weight_transform(weight: np.ndarray) -> np.ndarray:
    c, oC = weight.shape
    new_w_arr = weight.transpose().reshape(oC, c, 1, 1)
    return new_w_arr


def _gemm_weight_transfrom(weight: np.ndarray, attrs: Dict) -> np.ndarray:
    if attrs['transB'] != 0:
        weight = weight.transpose()
    return _matmul_weight_transform(weight)


def _get_gemm_inputs(
    node: onnx.NodeProto, init_keys: AbstractSet[str]
) -> Tuple[str, str, Optional[str]]:
    assert node.op_type == "Gemm", repr(node)

    for node_input in node.input[:2]:
        if node_input in init_keys:
            weight_tensor = node_input
        else:
            input_tensor = node_input

    bias_tensor = node.input[2] if len(node.input) == 3 else None

    return input_tensor, weight_tensor, bias_tensor


def _get_input_index(input_tensor: str, node: onnx.NodeProto) -> int:
    return list(node.input).index(input_tensor)


def _needs_gemm_transpose(input_idx: int, attrs: Dict) -> bool:
    assert input_idx in (0, 1)
    return attrs['transA' if input_idx == 0 else 'transB'] != 0


def _get_gemm_attrs(node: onnx.NodeProto) -> Dict:
    assert node.op_type == "Gemm", repr(node)

    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

    return {
        'alpha': attrs.get('alpha', 1.0),
        'beta': attrs.get('beta', 1.0),
        'transA': attrs.get('transA', 0),
        'transB': attrs.get('transB', 0),
    }


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> MatMul --> Add --> next
    to
        prev --> Unsqueeze --> Conv --> Squeeze --> next

    if 1. MatMul.ndim == 2
       2. MatMul must have exactly one initializer
       3. Add must have exactly one initializer
       4. Add's input with initializer is multidirectional broadcastable to (1, oC)
    """

    pattern_to_match = ['MatMul', 'Add']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
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

        matmul = matched_nodes[0]
        return matmul.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        matmul, add = nodes_to_check
        return (
            self.check_condition_1(matmul)
            and self.check_condition_2(matmul)
            and self.check_condition_2(add)
            and self.check_condition_3(matmul, add)
        )

    def check_condition_1(self, node: onnx.NodeProto) -> bool:
        return len(self.get_value_info_shape(node.output[0])) == 2

    def check_condition_2(self, node: onnx.NodeProto) -> bool:
        return sum(node_input in self.initializer_map for node_input in node.input) == 1

    def check_condition_3(self, node: onnx.NodeProto, node_1: onnx.NodeProto) -> bool:
        bias_tensor = self.get_init_node_input(node_1)
        bias_arr = self.get_initializer_array(bias_tensor)
        oC = self.get_value_info_shape(node.output[0])[1]

        return _is_np_broadcastable(bias_arr, (1, oC))

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        matmul, add = matched_nodes
        matmul_data_tensor = self.get_data_node_input(matmul)
        matmul_init_tensor = self.get_init_node_input(matmul)
        add_init_tensor = self.get_init_node_input(add)

        unsqueeze = self.make_node(
            'Unsqueeze',
            inputs=[matmul_data_tensor],
            outputs=[matmul.output[0] + '_unsqueezed'],
            name=matmul.output[0] + '_1',
            axes=[2, 3],
        )

        conv = self.make_node(
            'Conv',
            inputs=[
                unsqueeze.output[0],
                matmul_init_tensor + '_fused',
                add_init_tensor + '_fused',
            ],
            outputs=[matmul.output[0] + '_fused'],
            name=matmul.output[0] + '_2',
        )

        squeeze_node = self.make_node(
            'Squeeze',
            inputs=[conv.output[0]],
            outputs=[add.output[0]],
            name=matmul.output[0] + '_3',
            axes=[2, 3],
        )
        return [unsqueeze, conv, squeeze_node]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        matmul, add = matched_nodes

        weight_tensor = self.get_init_node_input(matmul)
        w_arr = self.get_initializer_array(weight_tensor)
        new_w_arr = _matmul_weight_transform(w_arr)
        new_weight = self.make_initializer_from_array(new_w_arr, weight_tensor + '_fused')

        bias_tensor = self.get_init_node_input(add)
        bias_arr = self.get_initializer_array(bias_tensor)
        oC = self.get_value_info_shape(matmul.output[0])[1]
        new_bias_arr = np.broadcast_to(bias_arr, (1, oC)).flatten()
        new_bias = self.make_initializer_from_array(new_bias_arr, bias_tensor + '_fused')

        return [new_weight, new_bias]

    def make_new_vi(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        matmul, _ = matched_nodes
        matmul_data_tensor = self.get_data_node_input(matmul)

        conv_input_vi = self.make_tensor_value_info(
            matmul.output[0] + '_unsqueezed',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(matmul_data_tensor) + [1, 1],
        )

        conv_output_vi = self.make_tensor_value_info(
            matmul.output[0] + '_fused',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(matmul.output[0]) + [1, 1],
        )

        return [conv_input_vi, conv_output_vi]


class Pattern_2(ONNXTransformer):
    """
    transform
        prev --> Gemm --> next
    to
        prev --> Unsqueeze --> Conv --> Squeeze --> next
    if 1. Gemm.B must be defined in initializer whereas Gemm.A must not
       2. if Gemm.C is defined, Gemm.C must be an initializer and multidirectional broadcastable to (1, oC)
       3. all of Gemm.input must have onnx.TensorProto.FLOAT dtype
    """

    pattern_to_match = ['Gemm']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
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
        gemm = matched_nodes[0]
        return gemm.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        (gemm,) = nodes_to_check
        return (
            self.check_condition_1(gemm)
            and self.check_condition_2(gemm)
            and self.check_condition_3(gemm)
        )

    def check_condition_1(self, node: onnx.NodeProto) -> bool:
        return node.input[0] not in self.initializer_map and node.input[1] in self.initializer_map

    def check_condition_2(self, node: onnx.NodeProto) -> bool:
        if len(node.input) == 3:
            if node.input[2] in self.initializer_map:
                array = self.get_initializer_array(node.input[2])
                oC = self.get_value_info_shape(node.output[0])[1]

                return _is_np_broadcastable(array, (1, oC))
            # returns False since Gemm.C has no initializer to be fused
            return False
        # always returns True if Gemm.C is not defined.
        return True

    def check_condition_3(self, node: onnx.NodeProto) -> bool:
        return all(
            self.get_value_info_dtype(tensor) == onnx.TensorProto.FLOAT for tensor in node.input
        )

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        (gemm,) = matched_nodes
        input_tensor, weight_tensor, bias_tensor = _get_gemm_inputs(gemm, set(self.initializer_map))
        attrs = _get_gemm_attrs(gemm)

        new_nodes = []
        unsqueeze_input = input_tensor
        input_idx = _get_input_index(input_tensor, gemm)
        if _needs_gemm_transpose(input_idx, attrs):
            unsqueeze_input += '_transposed'
            transpose = self.make_node(
                'Transpose',
                inputs=[input_tensor],
                outputs=[unsqueeze_input],
                name=gemm.output[0] + '_0',
            )
            new_nodes.append(transpose)

        unsqueeze = self.make_node(
            'Unsqueeze',
            inputs=[unsqueeze_input],
            outputs=[gemm.output[0] + '_unsqueezed'],
            name=gemm.output[0] + '_1',
            axes=[2, 3],
        )

        conv_inputs = [unsqueeze.output[0], weight_tensor + '_fused']
        if bias_tensor is not None:
            conv_inputs.append(bias_tensor + '_fused')
        conv = self.make_node(
            'Conv',
            inputs=conv_inputs,
            outputs=[gemm.output[0] + '_fused'],
            name=gemm.output[0] + '_2',
        )

        squeeze = self.make_node(
            'Squeeze',
            inputs=[conv.output[0]],
            outputs=[gemm.output[0]],
            name=gemm.output[0] + '_3',
            axes=[2, 3],
        )

        new_nodes.extend([unsqueeze, conv, squeeze])
        return new_nodes

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        (gemm,) = matched_nodes
        _, weight_tensor, bias_tensor = _get_gemm_inputs(gemm, set(self.initializer_map))
        attrs = _get_gemm_attrs(gemm)

        new_inits = []
        w_arr = self.get_initializer_array(weight_tensor) * attrs['alpha']
        new_w_arr = _gemm_weight_transfrom(w_arr, attrs)
        new_w_init = self.make_initializer_from_array(new_w_arr, weight_tensor + '_fused')
        new_inits.append(new_w_init)

        if bias_tensor:
            b_arr = self.get_initializer_array(bias_tensor)
            oC = self.get_value_info_shape(gemm.output[0])[1]
            new_b_arr = np.broadcast_to(b_arr, (1, oC)).flatten() * attrs['beta']
            new_b_init = self.make_initializer_from_array(new_b_arr, bias_tensor + '_fused')
            new_inits.append(new_b_init)

        return new_inits

    def make_new_vi(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.ValueInfoProto]:
        (gemm,) = matched_nodes
        input_tensor, _, _ = _get_gemm_inputs(gemm, set(self.initializer_map))

        new_vis = []
        attrs = _get_gemm_attrs(gemm)
        input_idx = _get_input_index(input_tensor, gemm)
        if _needs_gemm_transpose(input_idx, attrs):
            transpose_output_vi = self.make_tensor_value_info(
                input_tensor + '_transposed',
                onnx.TensorProto.FLOAT,
                self.get_value_info_shape(input_tensor)[::-1],
            )
            new_vis.append(transpose_output_vi)

        conv_input_vi = self.make_tensor_value_info(
            gemm.output[0] + '_unsqueezed',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(input_tensor) + [1, 1],
        )

        conv_output_vi = self.make_tensor_value_info(
            gemm.output[0] + '_fused',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(gemm.output[0]) + [1, 1],
        )
        new_vis.extend([conv_input_vi, conv_output_vi])
        return new_vis


class Pattern_3(ONNXTransformer):
    """
    transform
        prev --> Conv --> Add --> next
    to
        prev --> Conv --> next
    if  1. len(Conv.input) == 2 or (len(Conv.input) == 3 and Conv.input[2] has initializer)
        2. Add has only one initializer
        3. Add's input with initializer is multidirectional broadcastable to (1, oC, 1, 1)
    """

    pattern_to_match = ['Conv', 'Add']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            inits_to_add=self.make_new_init(matched_nodes),
        )
        conv = matched_nodes[0]
        return conv.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        conv, add = nodes_to_check
        return (
            self.check_condition_1(conv)
            and self.check_condition_2(add)
            and self.check_condition_3(conv, add)
        )

    def check_condition_1(self, node: onnx.NodeProto) -> bool:
        return len(node.input) == 2 or node.input[2] in self.initializer_map

    def check_condition_2(self, node: onnx.NodeProto) -> bool:
        return sum(node_input in self.initializer_map for node_input in node.input) == 1

    def check_condition_3(self, node: onnx.NodeProto, node_1: onnx.NodeProto) -> bool:
        bias_tensor = self.get_init_node_input(node_1)
        bias_arr = self.get_initializer_array(bias_tensor)
        oC = self.get_value_info_shape(node.output[0])[1]

        return _is_np_broadcastable(bias_arr, (1, oC, 1, 1))

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        conv, add = matched_nodes
        return [
            self.make_node(
                'Conv',
                inputs=[*conv.input[:2], self.get_init_node_input(add) + '_fused'],
                outputs=[add.output[0]],
                name=conv.name,
                **{attr.name: onnx.helper.get_attribute_value(attr) for attr in conv.attribute},
            )
        ]

    def make_new_init(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
        conv, add = matched_nodes
        bias_tensor_name = self.get_init_node_input(add)
        bias_array = self.get_initializer_array(bias_tensor_name)
        oC = self.get_value_info_shape(conv.output[0])[1]
        bias_array = np.broadcast_to(bias_array, (1, oC, 1, 1)).flatten()
        if len(conv.input) == 3:
            bias_array += self.get_initializer_array(conv.input[2]).flatten()
        return [self.make_initializer_from_array(bias_array, bias_tensor_name + '_fused')]


# TODO: fuse Conv + Mul into Conv
# class Pattern_4(Pattern_1):
#    """
#    transform
#        prev --> Conv --> Mul --> next
#    to
#        prev --> Conv --> next
#    if ...
#    """
