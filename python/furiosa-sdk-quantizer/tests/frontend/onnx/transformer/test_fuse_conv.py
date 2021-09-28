from abc import ABC

import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_conv import FuseConv
from tests.frontend.onnx.transformer import TestTransformer


# TODO 1. Generate test model that does not meet conditions for conv fusion
# TODO 2. Generate MatMul + Add test model directly by onnx
class UnitTestModel(nn.Module, ABC):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel, self).__init__()
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class UnitTestModel1(nn.Module, ABC):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel1, self).__init__()
        self.linear = nn.Linear(in_features=in_channel, out_features=out_channel, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class UnitTestModel2(nn.Module, ABC):
    """
        This creates Conv + Add graph for testing Pattern_3
    """

    def __init__(self, in_channel, out_channel):
        super(UnitTestModel2, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.add(x, torch.ones((1, x.shape[1], 1, 1)))
        return x


class MultiTestModel(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.linear(x)
        x = torch.add(x, torch.ones(x.shape))
        x = torch.add(x, torch.ones(x.shape))
        return x


class MultiTestModel1(UnitTestModel1):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel1, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.linear(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class TestFuseConv(TestTransformer, ABC):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(torch_model,
                                                       FuseConv(),
                                                       input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(8, 16)]
        in_channel = 16
        out_channel = 4

        op_types = ['Unsqueeze', 'Conv', 'Squeeze']

        orig_model, trans_model = self._make_test_model(UnitTestModel(in_channel, out_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(8, 16)]
        in_channel = 16
        out_channel = 4

        op_types = ['Unsqueeze', 'Conv', 'Squeeze']

        orig_model, trans_model = self._make_test_model(UnitTestModel1(in_channel, out_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(8, 16)]
        in_channel = 16
        out_channel = 4

        op_types = ['Mul', 'Unsqueeze', 'Conv', 'Squeeze', 'Add']

        orig_model, trans_model = self._make_test_model(MultiTestModel(in_channel, out_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shapes = [(8, 16)]
        in_channel = 16
        out_channel = 4

        op_types = ['Mul', 'Unsqueeze', 'Conv', 'Squeeze', 'Add']

        orig_model, trans_model = self._make_test_model(MultiTestModel1(in_channel, out_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5(self):
        """
            This tests Pattern_3
        """
        input_shapes = [(1, 16, 8, 8)]
        in_channel = 16
        out_channel = 4

        op_types = ['Conv']

        orig_model, trans_model = self._make_test_model(UnitTestModel2(in_channel, out_channel), input_shapes)

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
