from abc import ABC

import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_redundant_reshape_pattern import (
    FuseRedundantReshapePattern,
    Pattern_1,
    Pattern_2,
    Pattern_3,
)
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, ABC):
    def forward(self, x):
        x = x.reshape(shape=(4, 4, 8))
        x = x.reshape(shape=(8, 2, 8))
        return x


class UnitTestModel1(nn.Module, ABC):
    def forward(self, x):
        x = x.reshape(shape=(4, 4, 8))
        x = x.reshape(shape=(8, 2, 8))
        x = x.reshape(shape=(32, 4))
        return x


class UnitTestModel2(nn.Module, ABC):
    def forward(self, x):
        x = x.reshape(shape=(4, 4, 8))
        x = x.reshape(shape=(8, 2, 8))
        x = x.reshape(shape=(32, 4))
        x = x.reshape(shape=(4, 4, 2, 4))
        return x


class UnitTestModel3(nn.Module, ABC):
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = torch.unsqueeze(x, dim=0)
        return x


class TestFuseRedundantReshapePattern(TestTransformer, ABC):
    def _make_test_model(self, torch_model, input_shapes, transformer):
        orig_model, trans_model = self.make_test_model(torch_model, transformer, input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        input_shapes = [(16, 8)]

        op_types = ['Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel(), input_shapes, Pattern_1)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shapes = [(16, 8)]

        op_types = ['Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel1(), input_shapes, Pattern_2)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shapes = [(16, 8)]

        op_types = ['Reshape']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel2(), input_shapes, FuseRedundantReshapePattern()
        )
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shapes = [(16, 1, 8)]

        op_types = ['Reshape']

        orig_model, trans_model = self._make_test_model(UnitTestModel3(), input_shapes, Pattern_3)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
