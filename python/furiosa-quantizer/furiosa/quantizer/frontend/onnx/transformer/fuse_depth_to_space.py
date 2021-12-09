import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class FuseDepthToSpace(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Reshape --> Transpose --> Reshape --> next
    to
        prev --> DepthToSpace --> next

    if Transpose.perm == [0, 1, 4, 2, 5, 3] or == [0, 3, 4, 1, 5, 2]
    """

    def pattern_matching(self, base_node):
        inputs = base_node.input

        pattern_to_match = ['Reshape', 'Transpose', 'Reshape']
        matched_nodes = self.pattern_matcher(base_node, pattern_to_match)
        if not matched_nodes:
            return inputs

        if not self.pattern_condition_checker(matched_nodes):
            return inputs

        top_node, mid_node, _ = matched_nodes
        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=[
                self.make_node(
                    'DepthToSpace',
                    [top_node.input[0]],
                    [base_node.output[0]],
                    top_node.name,
                    **self.get_attrs(top_node, mid_node),
                )
            ],
        )

        return top_node.input

    def pattern_condition_checker(self, nodes_to_check):
        _, mid_node, _ = nodes_to_check
        perm = mid_node.attribute[0].ints
        if perm == [0, 1, 4, 2, 5, 3] or perm == [0, 3, 4, 1, 5, 2]:
            return True
        return False

    def get_attrs(self, top_node, mid_node):
        permutation = mid_node.attribute[0].ints
        reshaped_shape = self.get_value_info_shape(top_node.output[0])
        if all(x == y for (x, y) in zip(permutation, [0, 1, 4, 2, 5, 3])):
            mode = 'CRD'
            blocksize = reshaped_shape[2]
        elif all(x == y for (x, y) in zip(permutation, [0, 3, 4, 1, 5, 2])):
            mode = 'DCR'
            blocksize = reshaped_shape[1]
        else:
            raise Exception()

        return {'blocksize': blocksize, 'mode': mode}
