import abc
import warnings

import onnx
import numpy as np

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import ONNXTransformer

default_conv_attrs = {
    'dilations': [1, 1],
    'group': 1,
    'kernel_shape': [1, 1],
    'pads': [0, 0, 0, 0],
    'strides': [1, 1]
}


class FuseBnIntoConv(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer, abc.ABC):
    """
        transform
            prev --> Conv --> BatchNormalization --> next
        to
            prev --> Conv --> next
    """
    pattern_to_match = ['Conv', 'BatchNormalization']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node = matched_nodes[0]

        self.transform_to_fuse(matched_nodes,
                               nodes_to_add=[*self.make_new_node(matched_nodes)],
                               inits_to_add=[*self.make_new_init(matched_nodes)],
                               vis_to_add=[*self.make_new_vi(matched_nodes)] if self.make_new_vi(
                                   matched_nodes) else None
                               )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        return True

    def make_new_node(self, matched_nodes):
        top_node, base_node = matched_nodes

        input_names = [node_input if node_input not in self.initializer_map else node_input + '_bn_fused'
                       for node_input in top_node.input]

        return self.make_node('Conv', [*input_names], [base_node.output[0]], top_node.name,
                              **default_conv_attrs)

    def make_new_init(self, matched_nodes):
        top_node, base_node = matched_nodes
        bn_params = self.get_bn_params(base_node)
        multiplier, shifter = self.get_multiplier_and_shifter(*bn_params)

        inits_to_add = []
        for node_input in top_node.input:
            if node_input not in self.initializer_map:
                continue
            weight = self.get_initializer_array(node_input)
            fused_weight = self.fuse_bn_params(weight, multiplier, shifter)
            inits_to_add.append(self.make_initializer_from_array(fused_weight, node_input + '_bn_fused'))

        return inits_to_add

    def make_new_vi(self, matched_nodes):
        return None

    def get_bn_params(self, node):
        scale = self.get_initializer_array(node.input[1])
        if all(v == 0. for v in scale):
            warnings.warn(f'BatchNormalization.scale is a zero tensor: {node.input[1]}')

        B = self.get_initializer_array(node.input[2])
        mean = self.get_initializer_array(node.input[3])
        var = self.get_initializer_array(node.input[4])

        from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import attribute_to_kwargs
        attrs = attribute_to_kwargs(node.attribute)
        eps = attrs.get('epsilon', 1e-05)

        return scale, B, mean, var, eps

    @staticmethod
    def get_multiplier_and_shifter(scale, B, mean, var, eps):
        multiplier = scale * 1 / np.sqrt(var + eps)
        shifter = - mean * scale + B

        return multiplier, shifter

    @staticmethod
    def fuse_bn_params(weight, multiplier, shifter):
        if weight.ndim == 4:
            fused_weight = weight * multiplier.reshape(-1, 1, 1, 1)
            return fused_weight
        elif weight.ndim == 1:
            fused_bias = weight * multiplier + shifter
            return fused_bias
        else:
            raise Exception('Unknown weight ndim: %s' % weight.dim)


class Pattern_2(Pattern_1, abc.ABC):
    """
        transform
            prev --> BatchNormalization --> next
        to
            prev --> Mul --> Add --> next

        if prev.op_type != Conv
    """
    pattern_to_match = ['BatchNormalization']

    def pattern_condition_checker(self, nodes_to_check):
        node = nodes_to_check[0]

        if self.is_op_type(node.op_type, ['Conv']):
            return False
        return True

    def make_new_node(self, matched_nodes):
        node = matched_nodes[0]
        return [
            self.make_node('Mul', [node.input[0], node.input[0] + '_bn_multiplier'],
                           [node.output[0] + '_bn_multiplied'], node.name),
            self.make_node('Add',
                           [node.output[0] + '_bn_multiplied', node.input[0] + '_bn_shifter'],
                           [node.output[0]], node.name)
        ]

    def make_new_init(self, matched_nodes):
        node = matched_nodes[0]
        bn_params = self.get_bn_params(node)
        multiplier, shifter = self.get_multiplier_and_shifter(*bn_params)
        num_features = self.get_value_info_shape(node.output[0])[0]
        return [
            self.make_initializer_from_array(multiplier.reshape(num_features, -1, 1, 1),
                                             name=node.input[0] + '_bn_multiplier'),
            self.make_initializer_from_array(shifter.reshape(num_features, -1, 1, 1),
                                             name=node.input[0] + '_bn_shifter')
        ]

    def make_new_vi(self, matched_nodes):
        node = matched_nodes[0]
        return [self.make_tensor_value_info(node.output[0] + '_bn_multiplied',
                                            onnx.TensorProto.FLOAT,
                                            shape=self.get_value_info_shape(node.output[0]))]
