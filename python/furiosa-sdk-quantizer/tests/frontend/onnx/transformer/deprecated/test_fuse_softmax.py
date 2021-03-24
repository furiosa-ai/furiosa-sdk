from abc import ABC

import torch
import torch.nn as nn

from quantizer.frontend.onnx.transformer.deprecated.fuse_softmax import FuseSoftmax

from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, ABC):
    def __init__(self, dim):
        super(UnitTestModel, self).__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.exp(x)
        x = torch.sum(y, dim=self.dim, keepdim=True)
        x = torch.div(y, x)
        return x


class MultiTestModel(UnitTestModel, ABC):
    def __init__(self, dim):
        super(MultiTestModel, self).__init__(dim)

    def forward(self, x):
        x = torch.sub(x, torch.ones(x.shape))
        y = torch.exp(x)
        x = torch.sum(y, dim=self.dim, keepdim=True)
        x = torch.div(y, x)
        x = torch.sub(x, torch.ones(x.shape))
        return x


class TestFuseSoftmax(TestTransformer):
    def make_unit_model(self, input_shapes, dim):
        orig_model, trans_model = self.make_test_model(UnitTestModel(dim), FuseSoftmax(), input_shapes)
        return orig_model, trans_model

    def make_multi_model(self, input_shapes, dim):
        orig_model, trans_model = self.make_test_model(MultiTestModel(dim), FuseSoftmax(), input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        """
            Test the case where dim = 1(intermediate dim) for unit operator model
        """
        input_shapes = [(3, 2)]
        dim = 0
        op_types = ['Transpose', 'Softmax', 'Transpose']

        orig_model, trans_model = self.make_unit_model(input_shapes, dim)
        # TODO output values are not identical why?
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute([1, 0], trans_model.graph.node[0].attribute[0].ints)
        self.check_attribute(len(input_shapes[0]) - 1, trans_model.graph.node[1].attribute[0].i)
        self.check_attribute([1, 0], trans_model.graph.node[2].attribute[0].ints)

    def test_case2(self):
        """
            Test the case where dim = 3(intermediate dim) for unit operator model
        """
        input_shapes = [(1, 2, 3, 4, 5)]
        dim = 3
        op_types = ['Transpose', 'Softmax', 'Transpose']

        orig_model, trans_model = self.make_unit_model(input_shapes, dim)
        self.check_graph_node(trans_model, op_types)
        # TODO output values are not identical why?
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute([0, 1, 2, 4, 3], trans_model.graph.node[0].attribute[0].ints)
        self.check_attribute(len(input_shapes[0]) - 1, trans_model.graph.node[1].attribute[0].i)
        self.check_attribute([0, 1, 2, 4, 3], trans_model.graph.node[2].attribute[0].ints)

    def test_case3(self):
        """
            Test the case where dim = 1(last dim) for unit operator model
        """
        input_shapes = [(12, 24, 3)]
        dim = -1
        op_types = ['Softmax']

        orig_model, trans_model = self.make_unit_model(input_shapes, dim)
        self.check_graph_node(trans_model, op_types)
        # TODO output values are not identical why?
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute(len(input_shapes[0]) - 1, trans_model.graph.node[0].attribute[0].i)
