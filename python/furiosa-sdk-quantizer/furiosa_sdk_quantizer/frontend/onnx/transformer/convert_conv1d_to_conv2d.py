import abc

import onnx
import numpy as np

from furiosa_sdk_quantizer.interfaces.transformer import Transformer
from furiosa_sdk_quantizer.frontend.onnx.transformer import ONNXTransformer


class ConvertConv1dToConv2d(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer, abc.ABC):
    """
    transform
        prev --> Reshape --> Conv --> Reshape --> next
    to
        prev --> Reshape --> Conv --> Reshape --> next

    if Conv.input[0].ndim == 3, i.e., if Conv1d
    """

    pattern_to_match = ["Reshape", "Conv", "Reshape"]

    def pattern_matching(self, base_node):
        inputs = base_node.input

        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node, mid_node, base_node = matched_nodes
        new_mid_input_shape = [*self.get_value_info_shape(mid_node.input[0]), 1]
        new_top_reshape_shape = [*self.get_initializer_array(top_node.input[1]), 1]
        new_mid_output_shape = [*self.get_value_info_shape(mid_node.output[0]), 1]
        new_mid_weight_shape = [*self.get_value_info_shape(mid_node.input[1]), 1]

        self.transform_to_convert(
            matched_nodes,
            nodes_to_add=[
                self.make_node(
                    "Reshape",
                    [top_node.input[0], top_node.input[1] + "_converted"],
                    [top_node.output[0]],
                    top_node.name,
                ),
                self.make_node(
                    "Conv",
                    [
                        mid_node.input[0],
                        mid_node.input[1] + "_converted",
                        mid_node.input[2] if len(mid_node.input) == 3 else None,
                    ],
                    [mid_node.output[0]],
                    mid_node.name,
                    **self.get_attrs(mid_node)
                ),
                base_node,
            ],
            inits_to_add=[
                self.make_initializer_from_array(
                    np.array(new_top_reshape_shape), name=top_node.input[1] + "_converted"
                ),
                self.make_initializer_from_array(
                    self.get_initializer_array(mid_node.input[1]).reshape(new_mid_weight_shape),
                    name=mid_node.input[1] + "_converted",
                ),
                self.initializer_map[mid_node.input[0]] if len(mid_node.input) == 3 else None,
            ],
            vis_to_add=[
                self.make_tensor_value_info(
                    mid_node.input[0], onnx.TensorProto.FLOAT, new_mid_input_shape
                ),
                self.make_tensor_value_info(
                    mid_node.output[0], onnx.TensorProto.FLOAT, new_mid_output_shape
                ),
            ],
        )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        _, mid_node, _ = nodes_to_check
        if len(self.get_value_info_shape(mid_node.input[0])) == 3:
            return True
        return False

    def get_attrs(self, mid_node):
        from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import attribute_to_kwargs

        attrs = attribute_to_kwargs(mid_node.attribute)
        dilations = attrs.get("dilations", [1])
        group = attrs.get("group", 1)
        kernel_shape = attrs["kernel_shape"]
        pads = attrs.get("pads", [0, 0])
        strides = attrs.get("strides", [1])

        return {
            "dilations": [*dilations, 1],
            "group": group,
            "kernel_shape": [*kernel_shape, 1],
            "pads": [pads[0], 0, pads[1], 0],
            "strides": [strides[0], 1],
        }
