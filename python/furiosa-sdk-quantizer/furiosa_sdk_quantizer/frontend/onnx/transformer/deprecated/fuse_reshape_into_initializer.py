import onnx

from onnx.helper import make_tensor
from onnx import numpy_helper
from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class FuseReshapeIntoInitializer(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        nodes_by_output_name = {node.output[0]: node for node in model.graph.node}
        initializer = {init.name: init for init in model.graph.initializer}
        initializer_key = initializer.keys()
        # assume Conv is followed by Mul(Conv --> Mul) & Mul takes one data input and one init input
        # a * (x * w + b) = (x * aw + ab)
        optimized_nodes = []
        removed_nodes = []
        for node in model.graph.node:
            for node_input in node.input:
                try:
                    prev_node = nodes_by_output_name[node_input]
                except KeyError:
                    optimized_nodes.append(node)
                    continue

                if prev_node.op_type != "Reshape":
                    optimized_nodes.append(node)
                    continue

                if prev_node.input[0] not in initializer_key:
                    optimized_nodes.append(node)
                    continue

                init = initializer[prev_node.input[0]]
                shape_init = initializer[prev_node.input[1]]
                init_arr = numpy_helper.to_array(initializer[prev_node.input[0]])
                shape_init_arr = numpy_helper.to_array(shape_init)
                reshaped_init_arr = init_arr.reshape(shape_init_arr)

                model.graph.initializer.append(
                    make_tensor(
                        name=node_input,
                        data_type=init.data_type,
                        dims=shape_init_arr,
                        vals=reshaped_init_arr.flatten(),
                    )
                )

                model.graph.initializer.remove(init)
                model.graph.initializer.remove(shape_init)
                removed_nodes.append(prev_node)

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
