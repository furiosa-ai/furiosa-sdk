from typing import Iterable, List

import numpy as np
import onnx
from onnx.helper import make_model
import onnxoptimizer

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer, utils
from furiosa.quantizer.frontend.onnx.transformer.convert_2d_sum_to_add import Convert2dSumToAdd
from furiosa.quantizer.frontend.onnx.utils.inference_shape import InferenceShape
from furiosa.quantizer.frontend.onnx.utils.version_checker import CheckVersion
from furiosa.quantizer.interfaces.transformer import Transformer


class EmbeddingBagPorting(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        dynamic_axes = {"batch_size": 2048}
        input_shapes = {
            input.name: [
                dynamic_axes.get(dim.dim_param, dim.dim_value)
                for dim in input.type.tensor_type.shape.dim
            ]
            for input in model.graph.input
        }
        model = CheckVersion().transform(model)
        model = utils.name_nodes(model)
        model = onnxoptimizer.optimize(model, passes=["extract_constant_to_initializer"])
        model = EmbeddingBagPattern(model).transform()
        model = Convert2dSumToAdd().transform(model)
        model = InferenceShape(model).inference_shape(input_shapes)
        return model


class EmbeddingBagPattern(ONNXTransformer):
    """
    Apply to the DLRM model only.
    Transform onnx-exported EmbeddingBag graph to be equivalent to torch.EmbeddingBag
    If not applied on DLRM model, onnx simplifier doesn't work because transformed embeddingbag can't handle empty bag.
    https://github.com/pytorch/pytorch/blob/55d479aca5e959c5f2fe3089e162db710bb7632d/torch/onnx/symbolic_opset11.py#L836-L902
    """

    def pattern_matching(self, base_node):
        if base_node.op_type != 'Loop':
            return base_node.input
        subgraph = onnx.helper.get_attribute_value(base_node.attribute[0])
        subgraph_producer_map = {
            node_output: node for node in subgraph.node for node_output in node.output
        }
        subgraph_base_nodes = []
        for output in subgraph.output:
            output_producer = subgraph_producer_map[output.name]
            if output_producer.op_type == 'ReduceSum':
                subgraph_base_nodes.append(output_producer)
            elif output_producer.op_type in ['ReduceMean', 'ReduceMax']:
                # TODO case when EmbeddingBag mode is mean or max
                raise Exception("EmbeddingBagPorting of mean or max mode is not implemented")

        if not subgraph_base_nodes or len(subgraph_base_nodes) > 1:
            return base_node.input

        subgraph_base_node = subgraph_base_nodes[0]

        # code style to prevent pylint 'too many boolean expressions in if statement' and 'too-many-return-statements' errors
        gather_0 = subgraph_producer_map.get(subgraph_base_node.input[0], None)

        slice_0 = (
            subgraph_producer_map.get(gather_0.input[1], None) if gather_0 is not None else None
        )
        unsqueeze_0 = (
            subgraph_producer_map.get(slice_0.input[1], None) if slice_0 is not None else None
        )
        gather_1 = (
            subgraph_producer_map.get(unsqueeze_0.input[0], None) if gather_0 is not None else None
        )
        unsqueeze_1 = (
            subgraph_producer_map.get(slice_0.input[2], None) if slice_0 is not None else None
        )
        gather_2 = (
            subgraph_producer_map.get(unsqueeze_1.input[0], None)
            if unsqueeze_1 is not None
            else None
        )
        gather_0_op_type = None if gather_0 is None else gather_0.op_type
        slice_0_op_type = (
            None if slice_0 is None or gather_0_op_type != 'Gather' else slice_0.op_type
        )
        unsqueeze_0_op_type = (
            None if unsqueeze_0 is None or slice_0_op_type != 'Slice' else unsqueeze_0.op_type
        )
        gather_1_op_type = (
            None if gather_1 is None or unsqueeze_0_op_type != 'Unsqueeze' else gather_1.op_type
        )
        unsqueeze_1_op_type = (
            None if unsqueeze_1 is None or gather_1_op_type != 'Gather' else unsqueeze_1.op_type
        )
        gather_2_op_type = (
            None if gather_2 is None or unsqueeze_1_op_type != 'Unsqueeze' else gather_2.op_type
        )
        if gather_2_op_type != 'Gather':
            return base_node.input

        matched_nodes = [
            subgraph_base_node,
            gather_0,
            slice_0,
            unsqueeze_0,
            gather_1,
            unsqueeze_1,
            gather_2,
        ]

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        subgraph_transformer = ONNXTransformer(make_model(subgraph), name_nodes=False)

        subgraph_transformer.transform_to_fuse(
            [subgraph_base_node],
            nodes_to_add=_make_subgraph_new_node(matched_nodes),
            inits_to_add=_make_subgraph_new_init(matched_nodes),
        )
        subgraph_transformer.build_optimized_model(subgraph_transformer.model, check=False)

        self.transform_to_fuse(
            [base_node],
            nodes_to_add=_make_new_node(base_node, subgraph_transformer.model.graph),
        )

        return base_node.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        (
            subgraph_base_node,
            gather_0,
            slice_0,
            unsqueeze_0,
            gather_1,
            unsqueeze_1,
            gather_2,
        ) = nodes_to_check

        return (
            _check_condition_1(subgraph_base_node)
            and _check_condition_2(gather_0)
            and self.check_condition_3(slice_0)
            and _check_condition_4(unsqueeze_0)
            and _check_condition_4(unsqueeze_1)
            and _check_condition_5(gather_1, gather_2)
        )

    def check_condition_3(self, node: onnx.NodeProto) -> bool:
        axes = self.get_initializer_array(node.input[3])

        return np.array_equal(axes, [0])


def _check_condition_1(node: onnx.NodeProto) -> bool:
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    axes = attrs.get("axes")
    keepdims = attrs.get("keepdims")

    return axes is not None and axes == [0] and keepdims is not None and keepdims == 0


def _check_condition_2(node: onnx.NodeProto) -> bool:
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    axis = attrs.get("axis")

    return axis is not None and axis == 0


def _check_condition_4(node: onnx.NodeProto) -> bool:
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    axes = attrs.get("axes")

    return axes is not None and axes == [0]


def _check_condition_5(node_1: onnx.NodeProto, node_2: onnx.NodeProto) -> bool:
    for node in [node_1, node_2]:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        axis = attrs.get("axis")
        if axis is None or axis != 0:
            return False
    return node_1.input[1] == node_2.input[1]


def _make_subgraph_new_init(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.TensorProto]:
    _, gather, *_ = matched_nodes
    return [
        onnx.numpy_helper.from_array(np.array([1]), gather.output[0] + '_indice_1'),
    ]


def _make_subgraph_new_node(matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
    subgraph_base_node, gather, *_ = matched_nodes
    return [
        onnx.helper.make_node(
            'Shape',
            inputs=[gather.output[0]],
            outputs=[gather.output[0] + '_shape'],
            name=gather.name + '_shape',
        ),
        onnx.helper.make_node(
            'Gather',
            inputs=[gather.output[0] + '_shape', gather.output[0] + '_indice_1'],
            outputs=[gather.output[0] + '_shape_1'],
            name=gather.name + '_get_shape_1',
        ),
        onnx.helper.make_node(
            'ConstantOfShape',
            inputs=[gather.output[0] + '_shape_1'],
            outputs=[gather.output[0] + '_zero_vec'],
            name=gather.name + '_zero_vec',
        ),
        onnx.helper.make_node(
            'Unsqueeze',
            inputs=[gather.output[0] + '_zero_vec'],
            outputs=[gather.output[0] + '_zero_vec_unsqueezed'],
            name=gather.name + '_zero_vec_unsqueeze',
            axes=[0],
        ),
        onnx.helper.make_node(
            'Concat',
            inputs=[gather.output[0], gather.output[0] + '_zero_vec_unsqueezed'],
            outputs=[gather.output[0] + '_nonzero'],
            name=gather.name + '_convert',
            axis=0,
        ),
        onnx.helper.make_node(
            subgraph_base_node.op_type,
            inputs=[gather.output[0] + '_nonzero'],
            outputs=[subgraph_base_node.output[0]],
            name=subgraph_base_node.name,
            **{
                attr.name: onnx.helper.get_attribute_value(attr)
                for attr in subgraph_base_node.attribute
            },
        ),
    ]


def _make_new_node(base_node: onnx.NodeProto, subgraph: onnx.GraphProto) -> List[onnx.TensorProto]:
    return [
        onnx.helper.make_node(
            base_node.op_type,
            inputs=base_node.input,
            outputs=base_node.output,
            name=base_node.name,
            **{"body": subgraph},
        )
    ]
