import onnx

import numpy as np
from onnx import numpy_helper
from onnx.helper import make_tensor_value_info

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class FuseScalarMulIntoConv(Transformer):
    """
    from: Conv -> Mul
    to: Conv
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        nodes_by_output_name = {
            node_output: node for node in model.graph.node for node_output in node.output
        }
        nodes_by_input_name = {
            node_input: node for node in model.graph.node for node_input in node.input
        }
        value_info = {
            vi.name: vi
            for vi in list(model.graph.value_info)
            + list(model.graph.input)
            + list(model.graph.output)
        }
        initializer = {init.name: init for init in model.graph.initializer}

        # assume Conv is followed by Mul(Conv --> Mul) & Mul takes one data input and one init input
        # a * (x * w + b) = (x * aw + ab)
        post_fix = "_scalar_mul_fused"
        optimized_nodes = []
        removed_nodes = []
        for node in model.graph.node:
            if node.op_type != "Mul":
                optimized_nodes.append(node)
                continue

            def _is_input_op_type(node_input, op_type):
                try:
                    return nodes_by_output_name[node_input].op_type == op_type
                except KeyError:
                    return False

            def _is_input_init(node_input, initializer_keys):
                return node_input in initializer_keys

            idx_conv = list(
                filter(lambda enum: _is_input_op_type(enum[1], "Conv"), enumerate(node.input))
            )
            idx_init = list(
                filter(
                    lambda enum: _is_input_init(enum[1], initializer.keys()), enumerate(node.input)
                )
            )

            # Expect one of the inputs is Exp and the other is ReduceSum
            if not idx_conv or not idx_init:
                optimized_nodes.append(node)
                continue

            idx_conv = idx_conv[0][0]
            idx_init = idx_init[0][0]

            prev_node = nodes_by_output_name[node.input[idx_conv]]
            mul_factor = numpy_helper.to_array(initializer[node.input[idx_init]])

            try:
                assert not mul_factor.shape
            except AssertionError:
                optimized_nodes.append(node)
                continue

            for idx, node_input in enumerate(prev_node.input):
                if node_input in initializer.keys():
                    w_init = initializer[node_input]
                    w_arr = numpy_helper.to_array(w_init)
                    fused_w_arr = mul_factor * w_arr
                    fused_w_init = numpy_helper.from_array(fused_w_arr, name=w_init.name + post_fix)
                    prev_node.input[idx] += post_fix

                    model.graph.initializer.remove(w_init)
                    model.graph.initializer.append(fused_w_init)
                    model.graph.input.append(
                        make_tensor_value_info(
                            name=fused_w_init.name,
                            elem_type=fused_w_init.data_type,
                            shape=fused_w_arr.shape,
                        )
                    )
                    model.graph.input.remove(value_info[w_init.name])

            # change next node's input name instead of prev nodes' output
            for nnode in model.graph.node:
                for idx, input in enumerate(nnode.input):
                    if input == node.output[0]:
                        nnode.input[idx] = prev_node.output[0]

            if node.output[0] in [vi.name for vi in model.graph.output]:
                model.graph.output.remove(value_info[node.output[0]])
                model.graph.output.append(value_info[prev_node.output[0]])

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
