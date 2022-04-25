from abc import ABC, abstractmethod
from typing import List

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
from furiosa.quantizer.interfaces.transformer import Transformer


class EliminateClipper(Transformer):
    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        for transformer in [
            Pattern_1,
            Pattern_2,
            Pattern_3,
            Pattern_4,
            Pattern_5,
            Pattern_6,
        ]:
            transformer = transformer(model)
            model = transformer.transform()
        return model


class ClipperElimination(ONNXTransformer, ABC):
    """
    This class contains methods commonly used in various EliminateClipper patterns

    Eliminate clippers if
        1. a pattern given should be matched.
        2. every consumer's output quantization parameters of the previous node of clipper should be mutually identical.
    """

    # EliminateClipper().transform() is called on QDQ graph.
    # QDQ graph will apply QuantizeLinear - DequantizeLinear to bias of Conv operators.
    # bias of QLinearConv operator should be i32, so QDQ graph will have i32 quantization information for bias.
    # However, QuantizeLinear operator of onnx are only defined on i8 quantization parmeter, so temporarily, QDQ graph is not valid.
    check_runnable = False

    @property
    @abstractmethod
    def pattern_to_match(self) -> List[str]:
        pass

    def pattern_matching(self, base_node):
        matched_nodes = self.pattern_matcher(base_node, self.pattern_to_match)
        if not matched_nodes:
            return base_node.input

        if not self.pattern_condition_checker(matched_nodes):
            return base_node.input

        # cut out DQ(dqlinear)-clip-Q1(qlinear1) in ...-op-Q-DQ-clip-Q1-(DQ1)
        # connect op-Q with DQ1
        op, qlinear, dqlinear, clip, qlinear1 = matched_nodes[-5:]

        self.transform_to_eliminate([dqlinear, clip, qlinear1], qlinear.output[0])
        self.replace_quant_params(qlinear, qlinear1)
        self.remove_clip_qdq(clip)

        return op.input

    def pattern_condition_checker(self, matched_nodes):
        *_, _, qlinear, _, clip, _ = matched_nodes
        return self.check_condition_1(clip) and self.check_condition_2(qlinear)

    def check_condition_1(self, clip):
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        if clip.op_type == "Relu":
            return True

        if len(clip.input) == 1:
            return True
        if len(clip.input) == 2:
            min_tensor = self.find_prev_node(self.find_prev_node(clip.input[1]).input[0]).input[0]
            return min_tensor in self.initializer_map
        if len(clip.input) == 3:
            min_tensor = self.find_prev_node(self.find_prev_node(clip.input[1]).input[0]).input[0]
            max_tensor = self.find_prev_node(self.find_prev_node(clip.input[2]).input[0]).input[0]
            return min_tensor in self.initializer_map and max_tensor in self.initializer_map
        raise ValueError(clip)

    def check_condition_2(self, qlinear):
        """
        We assume multiple dqlinears(DQs) might follow from qlinear(Q) like in ...-op-Q-DQ-clip-Q1-...
                                                                                      +-DQ_1-clip-Q1_1-...
                                                                                      +-      ...
        Logic below checks the equality of qlinear1s(Q1s)' quantization parameters.
        """
        dqlinears = self.consumer_map[qlinear.output[0]]
        scales = set()
        zero_points = set()
        for dqlinear in dqlinears:
            clip = self.consumer_map[dqlinear.output[0]][0]
            qlinear1 = self.consumer_map[clip.output[0]][0]
            assert qlinear1.op_type == "QuantizeLinear", repr(qlinear1)
            scales.add(float(self.get_initializer_array(qlinear1.input[1])))
            zero_points.add(int(self.get_initializer_array(qlinear1.input[2])))
        return len(scales) == len(zero_points) == 1

    def remove_clip_qdq(self, clip):
        assert clip.op_type in ("Clip", "Relu"), repr(clip)
        for tensor_name in clip.input:
            dqlinear = self.traverse_prev_node(tensor_name, ["DequantizeLinear"])
            qlinear = self.traverse_prev_node(dqlinear.input[0], ["QuantizeLinear"])
            # remove dq-q nodes only if qlinear.input[0] is contant
            if qlinear.input[0] in self.initializer_map:
                self.pop_single_optimizer_map(dqlinear)
                self.pop_single_optimizer_map(qlinear)

    def replace_quant_params(self, qlinear, qlinear1):
        assert qlinear.op_type == "QuantizeLinear", repr(qlinear)
        assert qlinear1.op_type == "QuantizeLinear", repr(qlinear1)

        # replace qlinear's quant params with qlinear1's
        for idx in range(1, 3):
            old_tensor = qlinear.input[idx]
            new_tensor = qlinear1.input[idx]
            if np.array_equal(
                self.get_initializer_array(old_tensor), self.get_initializer_array(new_tensor)
            ):
                # if replacement already happened, then skip.
                continue
            self.initializer_map.pop(old_tensor)
            qlinear.input[idx] = new_tensor


class Pattern_1(ClipperElimination):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> next
    """

    pattern_to_match = ['Conv', 'QuantizeLinear', 'DequantizeLinear', 'Relu', 'QuantizeLinear']


class Pattern_2(ClipperElimination):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = ['Conv', 'QuantizeLinear', 'DequantizeLinear', 'Clip', 'QuantizeLinear']


class Pattern_3(ClipperElimination):
    """
    transform
        prev --> Add --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Add --> QuantizeLinear --> next
    """

    pattern_to_match = ['Add', 'QuantizeLinear', 'DequantizeLinear', 'Relu', 'QuantizeLinear']


class Pattern_4(ClipperElimination):
    """
    transform
        prev --> Add --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Add --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = ['Add', 'QuantizeLinear', 'DequantizeLinear', 'Clip', 'QuantizeLinear']


class Pattern_5(ClipperElimination):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> DequantizeLinear --> Relu --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> next
    """

    pattern_to_match = [
        'Conv',
        'QuantizeLinear',
        'DequantizeLinear',
        'Squeeze',
        'QuantizeLinear',
        'DequantizeLinear',
        'Relu',
        'QuantizeLinear',
    ]


class Pattern_6(ClipperElimination):
    """
    transform
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> DequantizeLinear --> Clip --> QuantizeLinear --> next
    to
        prev --> Conv --> QuantizeLinear --> DequantizeLinear --> Squeeze --> QuantizeLinear --> next

    if Clip.input[1], Clip.input[2]'s input before fake quantization have initializer
    """

    pattern_to_match = [
        'Conv',
        'QuantizeLinear',
        'DequantizeLinear',
        'Squeeze',
        'QuantizeLinear',
        'DequantizeLinear',
        'Clip',
        'QuantizeLinear',
    ]
