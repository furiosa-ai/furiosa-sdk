import abc

import torch
import torch.nn as nn

from furiosa_sdk_quantizer.frontend.onnx.transformer.fuse_gelu import FuseGELU

from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, abc.ABC):
    def forward(self, x):
        x = nn.functional.gelu(x)
        return x


class MultiTestModel(UnitTestModel, abc.ABC):
    def forward(self, x):
        x = torch.sub(x, torch.ones(x.shape))
        x = nn.functional.gelu(x)
        x = torch.sub(x, torch.ones(x.shape))
        return x


class TestFuseGELU(TestTransformer):
    def make_unit_model(self, input_shapes):
        orig_model, trans_model = self.make_test_model(UnitTestModel(), FuseGELU(), input_shapes)
        return orig_model, trans_model

    def make_multi_model(self, input_shapes):
        orig_model, trans_model = self.make_test_model(MultiTestModel(), FuseGELU(), input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        """
            Test whether the original model is well transformed for unit operator model,
            which contains only GELU operator
        """
        input_shapes = [(1, 1, 4, 4)]
        op_types = ['Gelu']

        orig_model, trans_model = self.make_unit_model(input_shapes)

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case2(self):
        """
            Test whether the original model is well transformed for multi operator model,
             which contains operators other than Gelu
        """
        input_shapes = [(2, 4, 8, 8, 16)]

        orig_model, trans_model = self.make_multi_model(input_shapes)

        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
