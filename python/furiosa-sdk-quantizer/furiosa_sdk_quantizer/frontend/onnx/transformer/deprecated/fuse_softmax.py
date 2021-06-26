import onnx
import numpy as np

from onnx.helper import make_node, make_tensor_value_info

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class FuseSoftmax(Transformer):
    """
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#softmax
    from: Exp -> ReduceSum -> Div
    to: Transpose -> Softmax -> Transpose

    Assume NCHW Input
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        initializer = {init.name: init for init in model.graph.initializer}
        value_info = {
            vi.name: vi
            for vi in list(model.graph.value_info)
            + list(model.graph.input)
            + list(model.graph.output)
        }

        post_fix = "_transposed"
        optimized_nodes = []
        removed_nodes = []
        for node in model.graph.node:
            if node.op_type != "Div":
                optimized_nodes.append(node)
                continue

            # Div has no specific order of input according to spec.
            # Therefore, we need to find the input index of Exp and ReduceSum.
            def _is_input_op_type(node_input, op_type):
                if node_input in initializer.keys():
                    return False
                return nodes_by_output_name[node_input].op_type == op_type

            idx_exp = list(
                filter(lambda enum: _is_input_op_type(enum[1], "Exp"), enumerate(node.input))
            )
            idx_rsum = list(
                filter(lambda enum: _is_input_op_type(enum[1], "ReduceSum"), enumerate(node.input))
            )

            # Expect one of the inputs is Exp and the other is ReduceSum
            if len(idx_exp) != 1 and len(idx_rsum) != 1:
                optimized_nodes.append(node)
                continue

            idx_exp = idx_exp[0][0]
            idx_rsum = idx_rsum[0][0]

            exp_node = nodes_by_output_name[node.input[idx_exp]]
            rsum_node = nodes_by_output_name[node.input[idx_rsum]]
            removed_nodes.extend([node, exp_node, rsum_node])

            # assert dim(input_shape) == 4
            exp_shape = [
                dim.dim_value for dim in value_info[exp_node.output[0]].type.tensor_type.shape.dim
            ]
            length = len(exp_shape)

            axis = rsum_node.attribute[0].ints

            # assert ReduceSum takes only 1 axis
            assert len(axis) == 1
            axis = axis[0]
            if axis == -1:
                axis = length - 1

            # make permutation according to axis given
            perm = list(range(0, length))
            perm[axis], perm[-1] = perm[-1], perm[axis]

            new_vi = []
            if axis != length - 1:
                trans_node_1 = make_node(
                    "Transpose",
                    inputs=[exp_node.input[0]],
                    outputs=[exp_node.output[0] + post_fix],
                    perm=perm,
                )

                softmax_node = make_node(
                    "Softmax",
                    inputs=[exp_node.output[0] + post_fix],
                    outputs=[exp_node.output[0] + "_softmax"],
                    axis=length - 1,
                )

                trans_node_2 = make_node(
                    "Transpose",
                    inputs=[exp_node.output[0] + "_softmax"],
                    outputs=[node.output[0]],
                    perm=perm,
                )
                optimized_nodes.extend([trans_node_1, softmax_node, trans_node_2])
                perm1_shape = np.array(exp_shape)[perm].tolist()
                new_vi.append(
                    make_tensor_value_info(
                        name=softmax_node.output[0],
                        elem_type=onnx.TensorProto.FLOAT,
                        shape=perm1_shape,
                    )
                )
                new_vi.append(
                    make_tensor_value_info(
                        name=trans_node_1.output[0],
                        elem_type=onnx.TensorProto.FLOAT,
                        shape=perm1_shape,
                    )
                )
            else:
                softmax_node = make_node(
                    "Softmax", inputs=[exp_node.input[0]], outputs=[node.output[0]], axis=length - 1
                )
                optimized_nodes.extend([softmax_node])

            model.graph.value_info.extend(new_vi)

        # remove duplicate node(s) in optimized nodes
        seen = []
        for op_node in optimized_nodes:
            if op_node in seen:
                continue
            seen.append(op_node)
        optimized_nodes = seen

        new_nodes = list(filter(lambda node: node not in removed_nodes, optimized_nodes))
        model = utils.rebuild_model(model, new_nodes)
        check_model(model)

        return model
