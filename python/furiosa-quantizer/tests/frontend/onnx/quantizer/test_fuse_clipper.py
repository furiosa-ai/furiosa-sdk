import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from furiosa.quantizer.frontend.onnx.transformer.polish_model import PolishModel
from tests import torch_to_onnx
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


class UnitTestModel4(nn.Module):
    """
    This creates Unsqueeze --> Unsqueeze --> Conv --> Squeeze --> Relu graph for testing Pattern_5
    """

    def __init__(self, num_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.clip = torch.clip

        self.dims = [2, 3]

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        x = self.conv(x)
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        x = nn.functional.relu(x)

        return x


class UnitTestModel5(UnitTestModel4):
    """
    This creates Unsqueeze --> Unsqueeze --> Conv --> Squeeze --> Clip graph for testing Pattern_6
    """

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        x = self.conv(x)
        x = torch.squeeze(x, -1)
        x = torch.squeeze(x, -1)
        x = torch.clip(x, 0.0, 1.0)

        return x


class TestFuseClipper(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model = torch_to_onnx(torch_model, input_shapes, torch.float32)
        orig_model = PolishModel().transform(orig_model)
        trans_model = post_training_quantization_with_random_calibration(
            model=orig_model, per_channel=True, static=True, mode=QuantizationMode.DFG, num_data=1
        )
        return orig_model, trans_model

    def test_case1(self):
        """
        This tests Pattern_1
        """
        input_shapes = [(2, 16, 4, 4)]
        in_channel = 16
        out_channel = 8

        op_types = ['QuantizeLinear', 'QLinearConv', 'DequantizeLinear']

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

        op_types = ['QuantizeLinear', 'QLinearConv', 'DequantizeLinear']

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

        op_types = [
            'QuantizeLinear',
            'DequantizeLinear',
            'DequantizeLinear',
            'Add',
            'QuantizeLinear',
            'DequantizeLinear',
        ]

        orig_model, trans_model = self._make_test_model(UnitTestModel2(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case4(self):
        """
        This tests Pattern_4
        """
        input_shapes = [(1, 3, 8, 8)]

        op_types = [
            'QuantizeLinear',
            'DequantizeLinear',
            'DequantizeLinear',
            'Add',
            'QuantizeLinear',
            'DequantizeLinear',
        ]

        orig_model, trans_model = self._make_test_model(UnitTestModel3(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case5(self):
        """
        This tests Pattern_5
        """
        num_channel = 32
        input_shapes = [(1, num_channel)]

        op_types = [
            'QuantizeLinear',
            'DequantizeLinear',
            'Unsqueeze',
            'QuantizeLinear',
            'DequantizeLinear',
            'Unsqueeze',
            'QuantizeLinear',
            'QLinearConv',
            'DequantizeLinear',
            'Squeeze',
            'QuantizeLinear',
            'DequantizeLinear',
        ]

        orig_model, trans_model = self._make_test_model(UnitTestModel4(num_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)

    def test_case6(self):
        """
        This tests Pattern_6
        """
        num_channel = 32
        input_shapes = [(1, num_channel)]

        op_types = [
            'QuantizeLinear',
            'DequantizeLinear',
            'Unsqueeze',
            'QuantizeLinear',
            'DequantizeLinear',
            'Unsqueeze',
            'QuantizeLinear',
            'QLinearConv',
            'DequantizeLinear',
            'Squeeze',
            'QuantizeLinear',
            'DequantizeLinear',
        ]

        orig_model, trans_model = self._make_test_model(UnitTestModel5(num_channel), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_value_info(trans_model)
