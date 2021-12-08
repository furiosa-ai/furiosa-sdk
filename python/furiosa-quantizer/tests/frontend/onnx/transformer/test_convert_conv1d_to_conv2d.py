import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.convert_conv1d_to_conv2d import (
    ConvertConv1dToConv2d,
)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnitTestModel, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False
        )

    def forward(self, x):
        x = x.reshape((1, 1, -1))
        x = self.conv(x)
        x = x.reshape((1, -1, 1, 1))
        return x


class TestConvertConv1dToConv2d(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(
            torch_model, ConvertConv1dToConv2d(), input_shapes
        )
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 16)]
        in_channel = 1
        out_channel = 1

        op_types = ['Reshape', 'Conv', 'Reshape']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel(in_channel, out_channel), input_shapes
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
