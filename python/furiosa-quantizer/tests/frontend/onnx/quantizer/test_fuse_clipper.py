import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.quantizer.fuse_clipper import FuseClipper
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    """
    This creates Conv + Relu graph for testing Pattern_1
    """

    def __init__(self, in_channel, out_channel):
        super(UnitTestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.clip = torch.clip

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class UnitTestModel1(UnitTestModel):
    """
    This creates Conv + Clip graph for testing Pattern_2
    """

    def forward(self, x):
        x = self.conv(x)
        x = self.clip(x, 0.0, 6.0)
        return x


class UnitTestModel2(nn.Module):
    """
    This creates Add + Relu graph for testing Pattern_3
    """

    def forward(self, x):
        x = torch.add(x, torch.ones(x.shape))
        x = nn.functional.relu(x)

        return x


class UnitTestModel3(nn.Module):
    """
    This creates Add + Clip graph for testing Pattern_4
    """

    def forward(self, x):
        x = torch.add(x, torch.ones(x.shape))
        x = torch.clip(x, -1.0, 1.0)

        return x


class TestFuseClipper(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(torch_model, FuseClipper(), input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        """
        This tests Pattern_1
        """
        input_shapes = [(2, 16, 4, 4)]
        in_channel = 16
        out_channel = 8

        op_types = ['Conv']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case2(self):
        """
        This tests Pattern_2
        """
        input_shapes = [(3, 6, 10, 10)]
        in_channel = 6
        out_channel = 9

        op_types = ['Conv']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel1(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case3(self):
        """
        This tests Pattern_3
        """
        input_shapes = [(1, 3, 8, 8)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel2(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case4(self):
        """
        This tests Pattern_4
        """
        input_shapes = [(1, 3, 8, 8)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel3(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)
