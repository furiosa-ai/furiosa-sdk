import onnx
from onnx import numpy_helper
from onnx.helper import make_model, make_node, make_tensor_value_info

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.transformer.polish_model import PolishModel
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model
from furiosa.quantizer.interfaces.transformer import Transformer


class ReifyConvForBert(Transformer):
    """
    from: MatMul + Add
    to: Conv

    Assume NCHW Input
    """

    def __init__(self):
        self.nodes_by_output_name = None
        self.initializers = None
        self.outputs_by_name = None

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        model = PolishModel().transform(model)

        self.nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        self.initializers = {init.name: init for init in model.graph.initializer}
        self.outputs_by_name = {oup.name: oup for oup in model.graph.output}

        model = self.transform_matmul_add(model)  # transform matmul + add --> conv
        check_model(model)

        return PolishModel().transform(model)

    def transform_matmul_add(self, model):
        optimized_nodes = []
        removed_nodes = []

        # Handle Case1: MatMul + Add
        for node in model.graph.node:
            if node.op_type != 'Add':
                optimized_nodes.append(node)
                continue

            # Add has no specific order of input according to spec.
            # Therefore, we need to find the input index of MatMul
            try:
                idx_matmul = [
                    i
                    for i, tensor_name in enumerate(node.input)
                    if tensor_name not in self.initializers
                    and self.nodes_by_output_name[tensor_name].op_type == 'MatMul'
                ]
            except KeyError:
                optimized_nodes.append(node)
                continue

            # Expect one of the inputs is Exp and the other is ReduceSum
            if len(idx_matmul) != 1:
                optimized_nodes.append(node)
                continue

            idx_matmul = idx_matmul[0]
            matmul_node = self.nodes_by_output_name[node.input[idx_matmul]]

            idx_matmul_init = next(
                i
                for i, tensor_name in enumerate(matmul_node.input)
                if tensor_name in self.initializers
            )

            matmul_init = self.initializers[matmul_node.input[idx_matmul_init]]
            matmul_weight = numpy_helper.to_array(matmul_init)
            c, n = matmul_weight.shape
            conv_weight = matmul_weight.transpose().reshape(n, c, 1, 1)

            idx_add_init = next(
                i
                for i, tensor_name in enumerate(matmul_node.input)
                if tensor_name in self.initializers
            )

            add_init = self.initializers[node.input[idx_add_init]]
            bias = numpy_helper.to_array(add_init)

            conv_weight_init = numpy_helper.from_array(
                conv_weight, name=matmul_node.input[idx_matmul_init] + '_reified'
            )
            bias_init = numpy_helper.from_array(bias, name=node.input[idx_add_init] + '_reified')
            model.graph.initializer.extend([conv_weight_init, bias_init])

            removed_nodes.extend([node, matmul_node])

            unsqueeze_node = make_node(
                'Unsqueeze',
                inputs=[matmul_node.input[0]],
                outputs=[matmul_node.output[0] + '_expanded'],
                **{
                    'axes': [
                        1,
                    ]
                },
            )

            transpose_node = make_node(
                'Transpose',
                inputs=[unsqueeze_node.output[0]],
                outputs=[matmul_node.output[0] + '_transposed'],
                **{'perm': [0, 3, 1, 2]},
            )

            conv_node = make_node(
                'Conv',
                inputs=[transpose_node.output[0], conv_weight_init.name, bias_init.name],
                outputs=[matmul_node.output[0] + '_conv_output'],
                **{
                    'dilations': [1, 1],
                    'group': 1,
                    'kernel_shape': [1, 1],
                    'pads': [0, 0, 0, 0],
                    'strides': [1, 1],
                },
            )

            squeeze_node = make_node(
                'Squeeze',
                inputs=[conv_node.output[0]],
                outputs=[matmul_node.output[0] + '_squeezed'],
                **{'axes': [2]},
            )

            transpose_node_1 = make_node(
                'Transpose',
                inputs=[squeeze_node.output[0]],
                outputs=[node.output[0]],
                **{'perm': [0, 2, 1]},
            )

            optimized_nodes.extend(
                [unsqueeze_node, transpose_node, conv_node, squeeze_node, transpose_node_1]
            )

            graph_input = model.graph.input[0]
            if conv_node.input[0] == graph_input.name:
                batch_size = graph_input.type.tensor_type.shape.dim[0].dim_value
                new_vi = make_tensor_value_info(
                    name=graph_input.name,
                    elem_type=graph_input.type.tensor_type.elem_type,
                    shape=(batch_size, c, 1, 1),
                )
                model.graph.input.remove(graph_input)
                model.graph.input.insert(0, new_vi)

        new_nodes = list(filter(lambda node: node not in removed_nodes, optimized_nodes))
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        model = make_model(model.graph)

        model = utils.eliminate_unused_protos(model)

        return model
