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


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> MatMul --> Add --> next
    to
        prev --> Unsqueeze --> Conv --> Squeeze --> next
    if 1. MatMul.ndim == 2
       2. MatMul must have exactly one initializer
       3. Add must have exactly one initializer
       4. Add initializer.ndim <= 2
          if Add initializer.ndim == 2, its batch_dim == 1
    """

    pattern_to_match = ['MatMul', 'Add']

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_nodes(
                **self.get_new_vi_args(matched_nodes), **self.get_new_init_args(matched_nodes)
            ),
            inits_to_add=self.make_initializers(**self.get_new_init_args(matched_nodes)),
            vis_to_add=self.make_value_infos(**self.get_new_vi_args(matched_nodes)),
        )

        top_node = matched_nodes[0]
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        matmul, add = nodes_to_check

        return (
            self.check_condition_1(matmul.output[0])
            and self.check_condition_2(matmul)
            and self.check_condition_2(add)
            and self.check_condition_3(add)
        )

    def check_condition_1(self, tensor_name):
        return len(self.get_value_info_shape(tensor_name)) == 2

    def check_condition_2(self, node):
        return (node.input[0] in self.initializer_map) != (node.input[1] in self.initializer_map)

    def check_condition_3(self, node):
        array = self.get_initializer_array(self.get_init_node_input(node))
        return (array.ndim == 2 and array.shape[0] == 1) or array.ndim == 1

    def get_new_vi_args(self, matched_nodes):
        matmul, add = matched_nodes

        return {
            'input_tensor_name': self.get_data_node_input(matmul),
            'output_tensor_name': add.output[0],
        }

    def get_new_init_args(self, matched_nodes):
        matmul, add = matched_nodes

        return {
            'weight_tensor_name': self.get_init_node_input(matmul),
            'bias_tensor_name': self.get_init_node_input(add),
        }

    def make_nodes(
        self, input_tensor_name, output_tensor_name, weight_tensor_name, bias_tensor_name
    ):
        unsqueeze = self.make_node(
            'Unsqueeze',
            inputs=[input_tensor_name],
            outputs=[output_tensor_name + '_unsqueezed'],
            name=input_tensor_name + '_1',
            axes=[2, 3],
        )

        conv = self.make_node(
            'Conv',
            inputs=[
                unsqueeze.output[0],
                weight_tensor_name + '_fused',
                bias_tensor_name + '_fused',
            ],
            outputs=[output_tensor_name + '_fused'],
            name=input_tensor_name + '_2',
        )

        squeeze = self.make_node(
            'Squeeze',
            inputs=[conv.output[0]],
            outputs=[output_tensor_name],
            name=input_tensor_name + '_3',
            axes=[2, 3],
        )

        return [unsqueeze, conv, squeeze]

    def make_initializers(self, weight_tensor_name, bias_tensor_name=None, **kwargs):
        new_inits = []
        weight_array = self.get_initializer_array(weight_tensor_name)
        new_weight_array = self.weight_transformation(weight_array, **kwargs)
        new_weight_init = self.make_initializer_from_array(
            new_weight_array, weight_tensor_name + '_fused'
        )
        new_inits.append(new_weight_init)

        if bias_tensor_name is not None:
            bias_array = self.get_initializer_array(bias_tensor_name).flatten()
            new_bias_init = self.make_initializer_from_array(
                bias_array, bias_tensor_name + '_fused'
            )
            new_inits.append(new_bias_init)

        return new_inits

    def weight_transformation(self, weight_array, **kwargs):
        c, n = weight_array.shape
        return weight_array.transpose().reshape(n, c, 1, 1)

    def make_value_infos(self, input_tensor_name, output_tensor_name):

        conv_input_vi = self.make_tensor_value_info(
            output_tensor_name + '_unsqueezed',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(input_tensor_name) + [1, 1],
        )

        conv_output_vi = self.make_tensor_value_info(
            output_tensor_name + '_fused',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(output_tensor_name) + [1, 1],
        )

        return [conv_input_vi, conv_output_vi]


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Gemm --> next
    to
        prev --> Unsqueeze --> Conv --> Squeeze --> next
    if 1. Gemm.B must be defined in initializer whereas Gemm.A must not
       2. if Gemm.C is defined, Gemm.C must be an initializer and Gemm.C.ndim <=2, especially, Gemm.C.shape[0] == 1 for Gemm.C.ndim == 2
       3. all of Gemm.input must have onnx.TensorProto.FLOAT dtype
    """

    pattern_to_match = ['Gemm']

    def pattern_condition_checker(self, nodes_to_check):
        (gemm,) = nodes_to_check
        return (
            self.check_condition_1(gemm)
            and self.check_condition_2(gemm)
            and self.check_condition_3(gemm)
        )

    def check_condition_1(self, node):
        return node.input[0] not in self.initializer_map and node.input[1] in self.initializer_map

    def check_condition_2(self, node):
        if len(node.input) == 3:
            if node.input[2] in self.initializer_map:
                array = self.get_initializer_array(node.input[2])
                return array.ndim == 2 and array.shape[0] == 1 or array.ndim == 1
            # returns False since Gemm.C has no initializer to be fused
            return False
        # always returns True if Gemm.C is not defined.
        return True

    def check_condition_3(self, node):
        return all(
            self.get_value_info_dtype(tensor) == onnx.TensorProto.FLOAT for tensor in node.input
        )

    def get_new_init_args(self, matched_nodes):
        (gemm,) = matched_nodes
        return {
            'weight_tensor_name': gemm.input[1],
            'bias_tensor_name': (gemm.input[2] if len(gemm.input) == 3 else None),
            **self.get_attrs(gemm),
        }

    def get_new_vi_args(self, matched_nodes):
        (gemm,) = matched_nodes
        return {
            'input_tensor_name': self.get_data_node_input(gemm),
            'output_tensor_name': gemm.output[0],
            'attrs': self.get_attrs(gemm),
        }

    def make_nodes(
        self,
        input_tensor_name,
        output_tensor_name,
        weight_tensor_name,
        bias_tensor_name=None,
        **kwargs,
    ):
        new_nodes = []
        unsqueeze_node_input = input_tensor_name
        if self.need_transpose(input_tensor_name, kwargs):
            unsqueeze_node_input = input_tensor_name + '_transposed'
            transpose_node = self.make_node(
                'Transpose',
                inputs=[input_tensor_name],
                outputs=[unsqueeze_node_input],
                name=output_tensor_name + '_0',
            )
            new_nodes.append(transpose_node)

        unsqueeze_node = self.make_node(
            'Unsqueeze',
            inputs=[unsqueeze_node_input],
            outputs=[input_tensor_name + '_unsqueezed'],
            name=output_tensor_name + '_1',
            axes=[2, 3],
        )

        conv_inputs = [unsqueeze_node.output[0], weight_tensor_name + '_fused']
        if bias_tensor_name is not None:
            conv_inputs.append(bias_tensor_name + '_fused')

        conv_node = self.make_node(
            'Conv',
            conv_inputs,
            outputs=[input_tensor_name + '_fused'],
            name=output_tensor_name + '_2',
        )

        squeeze_node = self.make_node(
            'Squeeze',
            inputs=[conv_node.output[0]],
            outputs=[output_tensor_name],
            name=output_tensor_name + '_3',
            axes=[2, 3],
        )

        new_nodes.extend([unsqueeze_node, conv_node, squeeze_node])
        return new_nodes

    def make_initializers(self, weight_tensor_name, bias_tensor_name=None, **kwargs):
        new_inits = []
        weight_array = self.get_initializer_array(weight_tensor_name) * kwargs['alpha']

        transpose_conv_weight = not kwargs['transB']
        new_weight_array = self.weight_transformation(weight_array, transpose_conv_weight)
        new_inits.append(
            self.make_initializer_from_array(new_weight_array, weight_tensor_name + '_fused')
        )

        if bias_tensor_name is not None:
            bias_array = self.get_initializer_array(bias_tensor_name) * kwargs['beta']
            new_inits.append(
                self.make_initializer_from_array(bias_array, bias_tensor_name + '_fused')
            )

        return new_inits

    def make_value_infos(self, input_tensor_name, output_tensor_name, attrs):
        new_vis = []
        if self.need_transpose(input_tensor_name, attrs):
            transpose_output_vi = self.make_tensor_value_info(
                input_tensor_name + '_transposed',
                onnx.TensorProto.FLOAT,
                self.get_value_info_shape(input_tensor_name)[::-1],
            )
            new_vis.append(transpose_output_vi)

        conv_input_vi = self.make_tensor_value_info(
            input_tensor_name + '_unsqueezed',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(input_tensor_name) + [1, 1],
        )

        conv_output_vi = self.make_tensor_value_info(
            input_tensor_name + '_fused',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(output_tensor_name) + [1, 1],
        )
        new_vis.extend([conv_input_vi, conv_output_vi])
        return new_vis

    def weight_transformation(self, weight_array, need_transpose):
        if need_transpose:
            weight_array = weight_array.transpose()

        n, c = weight_array.shape

        return weight_array.reshape(n, c, 1, 1)

    def get_attrs(self, node):
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        return {
            'alpha': attrs.get('alpha', 1.0),
            'beta': attrs.get('beta', 1.0),
            'transA': attrs.get('transA', 0),
            'transB': attrs.get('transB', 0),
        }

    def need_transpose(self, input_tensor_name, attrs):
        input_idx = self.get_node_input_idx(input_tensor_name)
        assert input_idx in [0, 1]
        return attrs['transA' if input_idx == 0 else 'transB']


class Pattern_3(Pattern_1):
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

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_nodes(matched_nodes),
            inits_to_add=self.make_initializers(matched_nodes),
        )
        conv = matched_nodes[0]
        return conv.input

    def pattern_condition_checker(self, nodes_to_check):
        conv, add = nodes_to_check
        return (
            self.check_condition_1(conv)
            and self.check_condition_2(add)
            and self.check_condition_3(conv, add)
        )

    def check_condition_1(self, node):
        return len(node.input) == 2 or node.input[2] in self.initializer_map

    def check_condition_2(self, node):
        return sum(node_input in self.initializer_map for node_input in node.input) == 1

    def check_condition_3(self, conv, add):
        bias_tensor_name = self.get_init_node_input(add)
        bias_array = self.get_initializer_array(bias_tensor_name)
        oC = self.get_value_info_shape(conv.output[0])[1]
        try:
            np.broadcast_to(bias_array, (1, oC, 1, 1))
            return True
        except ValueError:
            return False

    def make_nodes(self, matched_nodes):
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

    def make_initializers(self, matched_nodes):
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
