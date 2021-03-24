from abc import ABC

import torch
import torch.nn as nn

from quantizer.frontend.onnx.transformer.convert_clip_attr_to_input import ConvertClipAttrToInput

from tests.frontend.onnx.transformer import TestTransformer, init_to_numpy
from tests import make_test_model


class UnitTestModel(nn.Module, ABC):
    def __init__(self, min=None, max=None):
        super(UnitTestModel, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        x = torch.clamp(x, self.min, self.max)
        return x


class MultiTestModel(UnitTestModel):
    def __init__(self, min=None, max=None):
        super(MultiTestModel, self).__init__(min, max)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = torch.clamp(x, self.min, self.max)
        x = torch.add(x, torch.ones(x.shape))
        return x


class TestConvertClipAttrToInput(TestTransformer):
    def make_unit_model(self, input_shapes, min=None, max=None):
        orig_model, trans_model = self.make_test_model(UnitTestModel(min, max), ConvertClipAttrToInput(), input_shapes)
        return orig_model, trans_model

    def make_multi_model(self, input_shapes, min=None, max=None):
        orig_model, trans_model = self.make_test_model(MultiTestModel(min, max), ConvertClipAttrToInput(), input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 1, 4, 4)]
        output_shapes = [(1, 1, 4, 4)]
        min = 0.01
        op_types = ['Clip']
        onnx_model, _ = make_test_model(input_shapes=input_shapes, output_shapes=output_shapes,
                                        attributes={'min': min}, op_type=op_types[0], check_model=False)
        orig_model, trans_model = self.make_test_unit_model_from_onnx(onnx_model, ConvertClipAttrToInput())
        import onnx
        onnx.save_model(orig_model, 'tmp.onnx')
        onnx.save_model(trans_model, 'tmp1.onnx')
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)
        self.check_attribute(orig_model.graph.node[0].attribute[0].f, init_to_numpy(trans_model, 'X_0_clip_min'))

    def test_case2(self):
        input_shapes = [(1, 1, 4, 4)]
        output_shapes = [(1, 1, 4, 4)]
        max = 1.01
        op_types = ['Clip']
        onnx_model, _ = make_test_model(input_shapes=input_shapes, output_shapes=output_shapes,
                                        attributes={'max': max}, op_type=op_types[0], check_model=False)
        orig_model, trans_model = self.make_test_unit_model_from_onnx(onnx_model, ConvertClipAttrToInput())

        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)
        self.check_attribute(orig_model.graph.node[0].attribute[0].f, init_to_numpy(trans_model, 'X_0_clip_max'))

    def test_case3(self):
        input_shapes = [(1, 1, 4, 4)]
        output_shapes = [(1, 1, 4, 4)]
        min = 1.0099999904632568
        max = 1.01
        op_types = ['Clip']
        onnx_model, _ = make_test_model(input_shapes=input_shapes, output_shapes=output_shapes,
                                        attributes={'max': max, 'min': min}, op_type=op_types[0], check_model=False)
        orig_model, trans_model = self.make_test_unit_model_from_onnx(onnx_model, ConvertClipAttrToInput())

        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)
        self.check_attribute(orig_model.graph.node[0].attribute[0].f, init_to_numpy(trans_model, 'X_0_clip_min'))
        self.check_attribute(orig_model.graph.node[0].attribute[1].f, init_to_numpy(trans_model, 'X_0_clip_max'))

    def test_case4(self):
        input_shapes = [(8, 8, 8)]
        op_types = ['Clip']
        min = 0.01
        max = None
        orig_model, trans_model = self.make_unit_model(input_shapes, min, max)
        self.check_graph_node(trans_model, op_types)
        self.check_attribute(init_to_numpy(orig_model, '5'), init_to_numpy(trans_model, '5'))

    def test_case5(self):
        input_shapes = [(8, 8, 8)]
        op_types = ['Mul', 'Clip', 'Add']
        min = 0.01
        max = 1.01
        orig_model, trans_model = self.make_multi_model(input_shapes, min, max)
        self.check_graph_node(trans_model, op_types)
        self.check_attribute(init_to_numpy(orig_model, '36'), init_to_numpy(trans_model, '36'))
        self.check_attribute(init_to_numpy(orig_model, '37'), init_to_numpy(trans_model, '37'))
