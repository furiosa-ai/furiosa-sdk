from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizer.frontend.onnx.transformer.fuse_pad import (Pattern_1, Pattern_2, FusePad)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, ABC):
    def __init__(self):
        super(UnitTestModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

    def forward(self, x):
        x = F.pad(x, [2, 1, 2, 1, 0, 0, 0, 0], value=float('-inf'))
        x = self.maxpool(x)
        x = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0], value=0.0)
        x = self.maxpool(x)

        return x


class UnitTestModel_1(nn.Module, ABC):
    def __init__(self):
        super(UnitTestModel_1, self).__init__()
        self.maxpool3d = nn.MaxPool3d(kernel_size=7, stride=3, padding=1, ceil_mode=False)

    def forward(self, x):
        x = torch.add(x, torch.ones_like(x))
        x = F.pad(x, [1, 2, 3, 3, 2, 1, 0, 0, 0, 0], value=float('-inf'))
        x = self.maxpool3d(x)
        return x


class UnitTestModel2(nn.Module, ABC):
    def __init__(self):
        super(UnitTestModel2, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=3, padding=0, ceil_mode=False, count_include_pad=False)
        self.avgpool_1 = nn.AvgPool2d(kernel_size=4, stride=3, padding=2, ceil_mode=False, count_include_pad=True)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=5, stride=3, padding=1, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0], value=0.0)
        x = self.avgpool(x)
        x = self.avgpool_1(x)
        x = F.pad(x, [2, 1, 1, 2, 0, 0, 0, 0], value=0.0)
        x = self.avgpool_2(x)

        return x


class UnitTestModel3(nn.Module, ABC):
    def forward(self, x):
        x = UnitTestModel()(x)
        x = UnitTestModel2()(x)

        return x


class TestFusePad(TestTransformer, ABC):
    def _make_test_model(self, torch_model, input_shapes, transformer):
        orig_model, trans_model = self.make_test_model(torch_model,
                                                       transformer,
                                                       input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 3, 16, 19)]

        op_types = ['MaxPool', 'Pad', 'MaxPool']

        orig_model, trans_model = self._make_test_model(UnitTestModel(), input_shapes, Pattern_1)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case1_1(self):
        input_shapes = [(1, 3, 14, 16, 17)]

        op_types = ['Add', 'MaxPool']

        orig_model, trans_model = self._make_test_model(UnitTestModel_1(), input_shapes, Pattern_1)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(1, 3, 13, 11)]

        op_types = ['AveragePool', 'AveragePool', 'Pad', 'AveragePool']

        orig_model, trans_model = self._make_test_model(UnitTestModel2(), input_shapes, Pattern_2)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(1, 3, 112, 112)]

        orig_model, trans_model = self._make_test_model(UnitTestModel3(), input_shapes, FusePad())
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
