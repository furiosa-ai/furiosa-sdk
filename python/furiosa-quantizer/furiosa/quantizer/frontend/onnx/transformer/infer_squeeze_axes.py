from typing import Iterable, List

import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


# TODO change name to express general objective of transformer.
class InferSqueezeAxes(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


# TODO case when model's opset > 12 and Squeeze's input axes does not exist
class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Squeeze (axes attribute is None) --> next
    to
        prev --> Squeeze (axes attribute is filled using input's value info) --> next

    if  1. model's opset < 13.
        2. axes attribute of Squeeze does not exist
        3. Squeeze.input[0] has shape info (graph input or shape inferred value info)
    """

    pattern_to_match = ['Squeeze']

    def pattern_matching(self, base_node: onnx.NodeProto) -> List[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
        )

        return base_node.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        opset = next((opset for opset in self.model.opset_import if not opset.domain), None)
        if opset is None or opset.version >= 13:
            return False
        (squeeze,) = nodes_to_check
        try:
            if not self.get_value_info_shape(squeeze.input[0]):
                return False
        except ValueError:
            return False
        return all(attr.name != "axes" for attr in squeeze.attribute)

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        (squeeze,) = matched_nodes
        input_shape = self.get_value_info_shape(squeeze.input[0])
        axes = [i for (i, dim_value) in enumerate(input_shape) if dim_value == 1]

        new_squeeze = self.make_node(
            'Squeeze',
            inputs=squeeze.input,
            outputs=squeeze.output,
            name=squeeze.name,
            axes=axes,
        )
        return [new_squeeze]
