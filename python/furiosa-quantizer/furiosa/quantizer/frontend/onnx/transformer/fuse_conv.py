import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


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
       2. MatMul must have at most one initializer
       3. Add must have at most one initializer
    """

    pattern_to_match = ['MatMul', 'Add']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node = matched_nodes[0]

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[*self.make_nodes(**self.get_new_node_args(matched_nodes))],
            inits_to_add=[*self.make_initializers(**self.get_new_init_args(matched_nodes))],
            vis_to_add=[*self.make_value_infos(**self.get_new_vi_args(matched_nodes))],
        )
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        top_node, base_node = nodes_to_check

        if not self.check_condition_1(top_node.output[0]):
            return False

        if not self.check_condition_2(top_node):
            return False

        if not self.check_condition_2(base_node):
            return False

    def check_condition_1(self, tensor_name):
        if len(self.get_value_info_shape(tensor_name)) == 2:
            return True
        return False

    def check_condition_2(self, node):
        num_init = 0
        for node_input in node.input:
            if node_input in self.initializer_map:
                num_init += 1

        if num_init == 1:
            return True
        return False

    def get_new_vi_args(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]

        fnode_input = self.get_data_node_input(top_node)
        fnode_output = base_node.output[0]

        return {'node_input': fnode_input, 'node_output': fnode_output}

    def get_new_init_args(self, matched_nodes):
        top_node = matched_nodes[0]
        base_node = matched_nodes[-1]

        fw_input = self.get_init_node_input(top_node)
        fb_input = self.get_init_node_input(base_node)

        return {'w_input': fw_input, 'b_input': fb_input}

    def get_new_node_args(self, matched_nodes):
        args = dict()

        args.update(self.get_new_vi_args(matched_nodes))
        args.update(self.get_new_init_args(matched_nodes))

        return args

    def make_nodes(self, node_input, node_output, w_input, b_input, **kwargs):
        unsqueeze_node = self.make_node(
            'Unsqueeze',
            inputs=[node_input],
            outputs=[node_input + '_unsqueezed'],
            name=node_input + '_1',
            **{'axes': [2, 3]},
        )

        conv_node = self.make_node(
            'Conv',
            inputs=[unsqueeze_node.output[0], w_input + '_fused', b_input + '_fused'],
            outputs=[node_input + '_fused'],
            name=node_input + '_2',
            **{
                'dilations': [1, 1],
                'group': 1,
                'kernel_shape': [1, 1],
                'pads': [0, 0, 0, 0],
                'strides': [1, 1],
            },
        )

        squeeze_node = self.make_node(
            'Squeeze',
            inputs=[conv_node.output[0]],
            outputs=[node_output],
            name=node_input + '_3',
            **{'axes': [2, 3]},
        )
        return unsqueeze_node, conv_node, squeeze_node

    def make_initializers(self, w_input, b_input=None, **kwargs):
        new_inits = []
        w_arr = self.get_initializer_array(w_input)
        new_w_arr = self.weight_transformation(w_arr, **kwargs)
        new_w_init = self.make_initializer_from_array(new_w_arr, w_input + '_fused')
        new_inits.append(new_w_init)

        if b_input:
            b_arr = self.get_initializer_array(b_input)
            new_b_init = self.make_initializer_from_array(b_arr, b_input + '_fused')
            new_inits.append(new_b_init)

        return new_inits

    def weight_transformation(self, w_arr, **kwargs):
        c, n = w_arr.shape
        new_w_arr = w_arr.transpose().reshape(n, c, 1, 1)
        return new_w_arr

    def make_value_infos(self, node_input, node_output):

        conv_input_vi = self.make_tensor_value_info(
            node_input + '_unsqueezed',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(node_input) + [1, 1],
        )

        conv_output_vi = self.make_tensor_value_info(
            node_input + '_fused',
            onnx.TensorProto.FLOAT,
            self.get_value_info_shape(node_output) + [1, 1],
        )

        return conv_input_vi, conv_output_vi


class Pattern_2(Pattern_1):
    """
    transform
        prev --> Gemm --> next
    to
        prev --> Unsqueeze --> Conv --> Squeeze --> next

    if 1. one of Gemm.A and Gemm.B must have initializer
       2. Gemm.C must have initializer if defined
    """

    pattern_to_match = ['Gemm']

    def pattern_condition_checker(self, nodes_to_check):
        node = nodes_to_check[0]
        if not self.check_condition_3(node):
            return False

        if not self.check_condition_4(node):
            return False

        return True

    def check_condition_3(self, node):
        num_init = 0
        for idx, node_input in enumerate(node.input):
            if idx == 2:
                break
            if node_input in self.initializer_map:
                num_init += 1

        if num_init == 1:
            return True
        return False

    def check_condition_4(self, node):
        if len(node.input) == 3:
            if node.input[2] not in self.initializer_map:
                return False
        return True

    def get_new_init_args(self, matched_nodes):
        node = matched_nodes[0]

        fw_input = node.input[1]
        fb_input = None
        if len(node.input) == 3:
            fb_input = node.input[2]

        args = {'w_input': fw_input, 'b_input': fb_input}
        args.update(self.get_attrs(node))

        return args

    def get_new_vi_args(self, matched_nodes):
        node = matched_nodes[0]
        fnode_input = node.input[0]
        fnode_output = node.output[0]

        return {'node_input': fnode_input, 'node_output': fnode_output}

    def weight_transformation(self, w_arr, **kwargs):
        transB = kwargs['transB']
        if transB == 0:
            w_arr = w_arr.transpose()

        n, c = w_arr.shape

        new_arr = w_arr.reshape(n, c, 1, 1)
        return new_arr

    def get_attrs(self, node):
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        alpha = attrs['alpha']
        beta = attrs['beta']
        assert alpha == beta == 1.0, "Assume alpha = beta = 1.0"

        transB = attrs['transB']

        return {'transB': transB}


class Pattern_3(ONNXTransformer):
    """
    transform
        prev --> Conv --> Add --> next
    to
        prev --> Conv --> next
    if len(Conv.input) == 2
    """

    pattern_to_match = ['Conv', 'Add']

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node, base_node = matched_nodes

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[self.make_nodes(*matched_nodes)],
            inits_to_add=[self.make_initializers(base_node)],
        )
        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        top_node, _ = nodes_to_check
        return len(top_node.input) == 2

    def make_nodes(self, top_node, base_node):
        conv_node = self.make_node(
            'Conv',
            inputs=[*top_node.input, self.get_init_node_input(base_node) + '_fused'],
            outputs=[base_node.output[0]],
            name=top_node.name,
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in top_node.attribute},
        )

        return conv_node

    def make_initializers(self, base_node):
        b_input = self.get_init_node_input(base_node)
        b_arr = self.get_initializer_array(b_input)
        new_b_init = self.make_initializer_from_array(b_arr.flatten(), b_input + '_fused')

        return new_b_init
