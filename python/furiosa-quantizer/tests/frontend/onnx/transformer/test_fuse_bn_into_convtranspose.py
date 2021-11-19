import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_bn_into_convtranspose import (
    FuseBnIntoConvTranspose,
)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel, self).__init__()
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


class UnitTestModel1(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(UnitTestModel1, self).__init__(in_channel, out_channel)

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
        x = self.convtranspose(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class MultiTestModel1(UnitTestModel):
    def __init__(self, in_channel, out_channel):
        super(MultiTestModel1, self).__init__(in_channel, out_channel)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.convtranspose(x)
        x = torch.nn.functional.relu(x)
        x = self.bn(x)
        x = torch.add(x, torch.ones(x.shape))
        return x


class TestFuseBNIntoConv(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(
            torch_model, FuseBnIntoConvTranspose(), input_shapes
        )
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4

        op_types = ['ConvTranspose']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4

        op_types = ['ConvTranspose', 'Relu', 'BatchNormalization']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel1(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(1, 4, 4, 4)]
        in_channel = 4
        out_channel = 8

        op_types = ['Mul', 'ConvTranspose', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shapes = [(1, 5, 4, 4)]
        in_channel = 5
        out_channel = 2

        op_types = ['Mul', 'ConvTranspose', 'Relu', 'BatchNormalization', 'Add']

        orig_model, trans_model = self._make_test_model(
            MultiTestModel1(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
