import numpy as np
import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.eliminate_redundant_shape_pattern import (
    EliminateRedundantShapePattern,
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
    Pattern_5,
    Pattern_6,
    Pattern_7,
    Pattern_8,
)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.add(x, torch.ones_like(x))
        return x


class UnitTestModel_1(nn.Module):
    def forward(self, x):
        x = torch.add(x, torch.ones_like(x))
        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, dim=1)
        return x


class UnitTestModel2(nn.Module):
    def forward(self, x):
        x = torch.add(x, torch.ones_like(x))
        y = torch.reshape(x, x.shape)
        y = torch.flatten(y, 1)
        y = torch.unsqueeze(y, dim=1)
        x = torch.reshape(x, x.shape)
        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, dim=1)
        return x, y


class UnitTestModel2_1(nn.Module):
    def forward(self, x):
        x = torch.div(x, torch.ones_like(x))
        y = torch.reshape(x, x.shape)
        y = torch.flatten(y, 1)
        y = torch.unsqueeze(y, dim=1)
        x = torch.reshape(x, x.shape)
        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, dim=1)
        x = torch.add(x, torch.ones_like(x))
        return x, y


class UnitTestModel3(nn.Module):
    def forward(self, x):
        x = torch.reshape(x, x.shape)
        x = torch.div(x, torch.ones_like(x))
        y = torch.reshape(x, [*x.shape[1:], x.shape[0]])
        return x, y


class UnitTestModel3_1(nn.Module):
    def forward(self, x):
        x = torch.reshape(x, x.shape)
        x = torch.div(x, torch.ones_like(x))
        y = torch.reshape(x, [*x.shape[1:], x.shape[0]])
        y = torch.reshape(y, [*x.shape[1:], x.shape[0]])
        return x, y


class UnitTestModel4(nn.Module):
    def forward(self, x):
        x_origin = x
        new_shape = int(np.prod(x_origin.shape))
        x = torch.reshape(x, [new_shape])
        x = x.expand([1, new_shape])
        x = x.expand([1, 1, new_shape])
        x = torch.reshape(x, x_origin.shape)
        x = torch.add(x, torch.ones_like(x))
        return x


class UnitTestModel5(nn.Module):
    def forward(self, x):
        x_origin = x
        new_shape = int(np.prod(x_origin.shape))
        x = torch.reshape(x, [new_shape])
        x = x.expand([1, new_shape])
        x = torch.reshape(x, x_origin.shape)
        x = torch.add(x, torch.ones_like(x))
        return x


class UnitTestModel5_1(nn.Module):
    def forward(self, x):
        x_origin = x
        new_shape = int(np.prod(x_origin.shape))
        x = torch.reshape(x, [new_shape])
        x = x.expand([1, new_shape])
        x = torch.reshape(x, [x_origin.shape[-1], *x_origin.shape[:-1]])
        return x


class UnitTestModel6(nn.Module):
    def forward(self, x):
        x_origin = x
        new_shape = int(np.prod(x_origin.shape[1:]))
        x = torch.add(x, torch.ones_like(x))
        x = torch.reshape(x, [x_origin.shape[0], 1, new_shape])
        x = torch.reshape(x, x_origin.shape)
        return x


class UnitTestModel7(nn.Module):
    def forward(self, x):
        x_origin = x
        new_shape = int(np.prod(x_origin.shape[1:]))
        x = torch.add(x, torch.ones_like(x))
        x = torch.reshape(x, [x_origin.shape[0], 1, new_shape])
        x = torch.reshape(x, [x_origin.shape[0], 1, *x_origin.shape[1:]])
        x = torch.reshape(x, x_origin.shape)
        return x


class CompoundTestModel8(nn.Module):
    def forward(self, x):
        x = UnitTestModel()(x)
        x = UnitTestModel_1()(x)
        x, a = UnitTestModel2()(x)
        x, b = UnitTestModel2_1()(x)
        x, c = UnitTestModel3()(x)
        x = UnitTestModel4()(a)
        a = UnitTestModel5()(x)
        x = UnitTestModel5_1()(b)
        b = UnitTestModel6()(x)
        x = UnitTestModel7()(c)
        return x, a, b, c


class UnitTestModel8(nn.Module):
    def __init__(self):
        super(UnitTestModel8, self).__init__()
        self.conv = nn.Conv2d(4, 4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.expand([*x.shape])
        x = self.conv(x)

        return x


class TestFuseRedundantShapePattern(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes, transformer):
        orig_model, trans_model = self.make_test_model(torch_model, transformer, input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(1, 1, 576)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel(), input_shapes, Pattern_1)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case1_1(self):
        input_shapes = [(1, 1, 576)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel_1(), input_shapes, Pattern_1)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(1, 1, 576)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel2(), input_shapes, Pattern_2)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2_1(self):
        input_shapes = [(1, 1, 576)]

        op_types = ['Div', 'Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel2_1(), input_shapes, Pattern_2)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(3, 9, 9)]

        op_types = ['Div', 'Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel3(), input_shapes, Pattern_3)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3_1(self):
        input_shapes = [(3, 9, 9)]

        op_types = ['Div', 'Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel3_1(), input_shapes, Pattern_3)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shapes = [(2, 3, 3)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel4(), input_shapes, Pattern_4)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5(self):
        input_shapes = [(2, 3, 3)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel5(), input_shapes, Pattern_5)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case5_1(self):
        input_shapes = [(2, 3, 3)]

        op_types = ['Reshape', 'Expand', 'Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel5_1(), input_shapes, Pattern_5)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case6(self):
        input_shapes = [(2, 3, 3)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel6(), input_shapes, Pattern_6)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case7(self):
        input_shapes = [(2, 3, 3)]

        op_types = ['Add']

        orig_model, trans_model = self._make_test_model(UnitTestModel7(), input_shapes, Pattern_7)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case8(self):
        input_shapes = [(1, 16, 24, 8)]

        orig_model, trans_model = self._make_test_model(
            CompoundTestModel8(), input_shapes, EliminateRedundantShapePattern()
        )
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case9(self):
        input_shapes = [(1, 4, 8, 8)]

        op_types = ['Conv', 'Conv']

        orig_model, trans_model = self._make_test_model(UnitTestModel8(), input_shapes, Pattern_8)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
