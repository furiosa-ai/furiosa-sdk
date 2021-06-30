from abc import ABC

import torch
import torch.nn as nn

import numpy as np

from furiosa_sdk_quantizer.frontend.onnx.transformer.deprecated.fuse_scalar_mul_into_conv import FuseScalarMulIntoConv
from tests.frontend.onnx.transformer import TestTransformer, init_to_numpy


class UnitTestModel(nn.Module, ABC):
    def __init__(self, in_channel, out_channel, mul):
        super(UnitTestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(2, 2))
        self.mul = mul

    def forward(self, x):
        x = self.conv(x)
        x = torch.mul(x, self.mul)
        return x


class UnitTestModel1(UnitTestModel):
    def __init__(self, in_channel, out_channel, mul):
        super(UnitTestModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(2, 2))
        self.mul = mul

    def forward(self, x):
        x = self.conv(x)
        x = torch.mul(x, torch.from_numpy(np.ones(x.shape).astype(np.float32) * self.mul))
        return x


class MultiTestModel(UnitTestModel):
    def __init__(self, in_channel, out_channel, mul):
        super(MultiTestModel, self).__init__(in_channel, out_channel, mul)

    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = self.conv(x)
        x = torch.mul(x, self.mul)
        x = torch.add(x, torch.ones(x.shape))
        return x


class TestFuseMulIntoConv(TestTransformer, ABC):
    def _make_test_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(torch_model,
                                                       FuseScalarMulIntoConv(),
                                                       input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 3, 4, 4)]
        in_channel = 3
        out_channel = 4
        mul = 0.2

        op_types = ['Conv']

        orig_model, trans_model = self._make_test_model(UnitTestModel(in_channel, out_channel, mul), input_shapes)

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

        weight, bias = [init_to_numpy(orig_model, name) for name in ['conv.weight', 'conv.bias']]
        fused_weight, fused_bias = [init_to_numpy(trans_model, name + '_scalar_mul_fused') for name in
                                    ['conv.weight', 'conv.bias']]
        self.check_initializer(weight, fused_weight / mul)
        self.check_initializer(bias, fused_bias / mul)

    def test_case2(self):
        input_shapes = [(1, 3, 8, 8)]
        in_channel = 3
        out_channel = 4
        mul = 0.2
        op_types = ['Mul', 'Conv', 'Add']

        orig_model, trans_model = self._make_test_model(MultiTestModel(in_channel, out_channel, mul), input_shapes)

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

        weight, bias = [init_to_numpy(orig_model, name) for name in ['conv.weight', 'conv.bias']]
        fused_weight, fused_bias = [init_to_numpy(trans_model, name + '_scalar_mul_fused') for name in
                                    ['conv.weight', 'conv.bias']]
        self.check_initializer(weight, fused_weight / mul)
        self.check_initializer(bias, fused_bias / mul)
