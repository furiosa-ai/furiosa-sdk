import onnx

from furiosa.quantizer.frontend.onnx.transformer import fuse_bn_into_conv
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseBnIntoConvTranspose(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [Pattern_1]:
            model = transformer(model).transform()

        return model


class Pattern_1(fuse_bn_into_conv.Pattern_1):
    """
    transform
        prev --> ConvTranspose --> BatchNormalization --> next
    to
        prev --> ConvTranspose --> next
    """

    pattern_to_match = ['ConvTranspose', 'BatchNormalization']

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

    @staticmethod
    def fuse_bn_params(weight, multiplier, shifter):
        if weight.ndim == 4:
            fused_weight = weight * multiplier.reshape(1, -1, 1, 1)
            return fused_weight
        elif weight.ndim == 1:
            fused_bias = weight * multiplier + shifter
            return fused_bias
        else:
            raise Exception('Unknown weight ndim: %s' % weight.dim)
