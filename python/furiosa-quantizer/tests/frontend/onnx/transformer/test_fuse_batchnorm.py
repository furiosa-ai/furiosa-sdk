from pathlib import Path

import onnx
import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_batchnorm import (
    FuseBatchNorm,
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.bn.weight = nn.Parameter(torch.ones(out_channel))
        self.bn.bias = nn.Parameter(torch.ones(out_channel))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class UnitTestModel1(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel1, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.bn(x)
        return x


class UnitTestModel2(UnitTestModel):
    """
    This creates Conv graph for testing Pattern_3
    """

    def __init__(self, in_channel, out_channel):
        super(UnitTestModel2, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mul(x, torch.ones((1, x.shape[1], 1, 1)))
        x = torch.add(x, torch.ones((1, x.shape[1], 1, 1)))
        return x


class UnitTestModel3(UnitTestModel):
    """
    This creates Conv + Mul + Add graph for testing Pattern_3
    """

    def __init__(self, in_channel, out_channel):
        super(UnitTestModel3, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mul(x, x)
        x = torch.add(x, x)
        return x


class UnitTestModel3_1(UnitTestModel):
    """
    This creates Conv + Mul + Add graph for testing Pattern_3
    """

    def __init__(self, in_channel, out_channel):
        super().__init__(in_channel, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = torch.mul(x, torch.ones((1, x.shape[1], 1, 1)))
        x = torch.add(x, x)
        return x


class UnitTestModel4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel4, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=(2, 2)
        )
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.bn.weight = nn.Parameter(torch.ones(out_channel) * 0.333)
        self.bn.bias = nn.Parameter(torch.ones(out_channel) * 0.254)

    def forward(self, x):
        x = self.convtranspose(x)
        x = self.bn(x)
        return x


class UnitTestModel5(UnitTestModel4):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel5, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = self.convtranspose(x)
        x = torch.nn.functional.relu(x)
        x = self.bn(x)
        return x


class MultiTestModel(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.conv(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class MultiTestModel1(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel1, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class MultiTestModel2(UnitTestModel4):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel2, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.convtranspose(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class MultiTestModel3(UnitTestModel4):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel3, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.convtranspose(x)
        x = torch.nn.functional.relu(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class TestFuseBatchNorm(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes, pattern):
        orig_model, trans_model = self.make_test_model(torch_model, pattern, input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 3, 4, 4)]

        op_types = ['Conv']

        orig_model = onnx.load(Path(__file__).resolve().parent / "conv_bn.onnx")
        trans_model = Pattern_1(orig_model).transform()

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4

        op_types = ['Conv', 'Relu', 'Mul', 'Add']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel1(in_channel, out_channel), input_shapes, Pattern_4
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(1, 4, 4, 4)]
        in_channel = 4
        out_channel = 8

        op_types = ['Mul', 'Conv', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel(in_channel, out_channel), input_shapes, FuseBatchNorm()
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shapes = [(1, 5, 4, 4)]
        in_channel = 5
        out_channel = 2

        op_types = ['Mul', 'Conv', 'Relu', 'Mul', 'Add', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel1(in_channel, out_channel), input_shapes, FuseBatchNorm()
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5(self):
        """
        This tests Pattern_3
        """
        input_shapes = [(1, 3, 8, 8)]
        in_channel = 3
        out_channel = 4

        op_types = ['Conv']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel2(in_channel, out_channel), input_shapes, Pattern_3
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5_1(self):
        """
        This tests Pattern_3
        """
        input_shapes = [(1, 3, 8, 8)]
        in_channel = 3
        out_channel = 4

        op_types = ['Conv', 'Mul', 'Add']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel3(in_channel, out_channel), input_shapes, Pattern_3
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5_2(self):
        input_shapes = [(1, 4, 4, 4)]
        in_channel = 4
        out_channel = 8

        op_types = ['Conv', 'Mul', 'Add']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel3_1(in_channel, out_channel), input_shapes, Pattern_3
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case6(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4

        op_types = ['ConvTranspose']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel4(in_channel, out_channel), input_shapes, Pattern_2
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case7(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4

        op_types = ['ConvTranspose', 'Relu', 'BatchNormalization']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel5(in_channel, out_channel), input_shapes, Pattern_2
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case8(self):
        input_shapes = [(1, 4, 4, 4)]
        in_channel = 4
        out_channel = 8

        op_types = ['Mul', 'ConvTranspose', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel2(in_channel, out_channel), input_shapes, FuseBatchNorm()
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case9(self):
        input_shapes = [(1, 5, 4, 4)]
        in_channel = 5
        out_channel = 2

        op_types = ['Mul', 'ConvTranspose', 'Relu', 'Mul', 'Add', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel3(in_channel, out_channel), input_shapes, FuseBatchNorm()
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
