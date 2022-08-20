from typing import Iterable, List

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class ConvertNegativePadsToSlice(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
        ]:
            model = transformer(model).transform()

        return model


class Pattern_1(ONNXTransformer):
    """
    transform
        prev --> Pad --> next
    to
        prev --> Slice --> Pad --> next
    if
        1. Pad's pads input is an initializer with negative value sum of which does not exceed input size
    """

    pattern_to_match = ['Pad']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        self.transform_to_fuse(
            matched_nodes,
            nodes_to_add=self.make_new_node(matched_nodes),
            **self.make_new_init_and_vi(matched_nodes),
        )

        return base_node.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        (pad,) = nodes_to_check
        if pad.input[1] not in self.initializer_map:
            return False

        pads = self.get_initializer_array(pad.input[1])
        if not any(pad < 0 for pad in pads):
            return False

        input_shape = self.get_value_info_shape(pad.input[0])
        pad_per_axis = list(zip(pads[: len(pads) // 2], pads[len(pads) // 2 :]))

        # If sum of negative pad value exceeds input size, output size is negative, which leads to invaiid Pad operator
        # If one of negative pad value equals input size, original onnx Pad outputs 'nan' intermittently
        # If one of negative pad value exceeds input size, Pad is invalid
        return all(
            input_size + min(pad[0], 0) + min(pad[1], 0) >= 0
            and input_size + min(pad[0], 0) > 0
            and input_size + min(pad[1], 0) > 0
            for (input_size, pad) in zip(input_shape, pad_per_axis)
        )

    def make_new_node(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        (pad,) = matched_nodes

        slice_ = onnx.helper.make_node(
            'Slice',
            inputs=[
                pad.input[0],
                pad.output[0] + '_starts',
                pad.output[0] + '_ends',
                pad.output[0] + '_axes',
            ],
            outputs=[pad.output[0] + '_slice_out'],
            name=pad.output[0] + '_0',
        )

        new_pad = onnx.helper.make_node(
            'Pad',
            inputs=[pad.output[0] + '_slice_out', pad.output[0] + '_pads', *pad.input[2:]],
            outputs=pad.output,
            name=pad.output[0] + '_1',
        )

        return [slice_, new_pad]

    def make_new_init_and_vi(self, matched_nodes: Iterable[onnx.NodeProto]) -> List[onnx.NodeProto]:
        (pad,) = matched_nodes
        input_shape = self.get_value_info_shape(pad.input[0])
        new_inits = []

        pads = self.get_initializer_array(pad.input[1])
        pad_per_axis = list(zip(pads[: len(pads) // 2], pads[len(pads) // 2 :]))
        starts, ends, axes = zip(
            *[
                (max(-pad[0], 0), input_shape[i] + min(pad[1], 0), i)
                for (i, pad) in enumerate(pad_per_axis)
                if pad[0] < 0 or pad[1] < 0
            ]
        )
        pads = [max(pad, 0) for pad in pads]
        for (init_list, suffix) in zip(
            (starts, ends, axes, pads), ('_starts', '_ends', '_axes', '_pads')
        ):
            new_init = onnx.numpy_helper.from_array(
                np.array(init_list, dtype=np.int64), pad.output[0] + suffix
            )
            new_inits.append(new_init)

        new_vis = []
        slice_output_shape = [
            int(shape + min(pad[0], 0) + min(pad[1], 0))
            for (shape, pad) in zip(input_shape, pad_per_axis)
        ]
        slice_output_vi = onnx.helper.make_tensor_value_info(
            pad.output[0] + '_slice_out',
            onnx.TensorProto.FLOAT,
            slice_output_shape,
        )
        new_vis.append(slice_output_vi)

        return {'inits_to_add': new_inits, 'vis_to_add': new_vis}


class Pattern_2(ONNXTransformer):
    """
    transform
        prev --> Pad --> next
    to
        prev --> next
    if
        1. Pad's pads value are all zero
    """

    pattern_to_match = ['Pad']

    def pattern_matching(self, base_node: onnx.NodeProto) -> Iterable[str]:
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        (pad,) = matched_nodes

        self.transform_to_eliminate(matched_nodes, pad.input[0])

        return base_node.input

    def pattern_condition_checker(self, nodes_to_check: Iterable[onnx.NodeProto]) -> bool:
        (pad,) = nodes_to_check
        if pad.input[1] not in self.initializer_map:
            return False

        pads = self.get_initializer_array(pad.input[1])
        return all(pad == 0 for pad in pads)
