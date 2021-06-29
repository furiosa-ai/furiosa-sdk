import onnx

from onnx.helper import make_node, make_tensor, TensorProto

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model


class ConvertClipAttrToInput(Transformer):
    """
    https://github.com/onnx/onnx/blob/master/docs/Operators.md#softmax
    from: [max, min] in node.attribute
    to: [min, max] in node.input

    Assume NCHW Input
    """

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        optimized_nodes = []
        for node in model.graph.node:
            if node.op_type != "Clip":
                optimized_nodes.append(node)
                continue

            if len(node.input) >= 2:
                optimized_nodes.append(node)
                continue

            node_input = node.input[0]
            node_output = node.output[0]

            input_names = dict()
            added_inits = dict()

            input_names["min"] = ""
            input_names["max"] = ""
            # The filter() method constructs an iterator from elements of an iterable for which a function returns true.
            for attr in filter(lambda x: x.name == "min" or x.name == "max", node.attribute):
                tensor_name = f"{node.input[0]}_clip_{attr.name}"
                tensor = make_tensor(tensor_name, TensorProto.FLOAT, (), [attr.f])
                input_names[attr.name] = tensor_name
                added_inits[attr.name] = tensor

            model.graph.initializer.extend([*added_inits.values()])

            new_node = make_node(
                "Clip",
                inputs=[node_input, input_names["min"], input_names["max"]],
                outputs=[node_output],
            )

            optimized_nodes.append(new_node)

        # remove duplicate node(s) in optimized nodes
        seen = []
        for op_node in optimized_nodes:
            if op_node in seen:
                continue
            seen.append(op_node)
        optimized_nodes = seen

        model = utils.rebuild_model(model, optimized_nodes)
        check_model(model)

        return model
