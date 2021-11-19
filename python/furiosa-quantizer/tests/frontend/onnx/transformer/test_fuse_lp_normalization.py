import abc
import unittest

import onnxruntime
import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_lp_normalization import FuseLpNormalization
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, abc.ABC):
    def __init__(self, p, dim):
        super(UnitTestModel, self).__init__()
        self.lp_normalization = nn.functional.normalize
        self.p = p
        self.dim = dim

    def forward(self, x):
        x = self.lp_normalization(x, p=self.p, dim=self.dim)
        return x


class MultiTestModel(UnitTestModel, abc.ABC):
    def __init__(self, p, dim):
        super(MultiTestModel, self).__init__(p, dim)

    def forward(self, x):
        x = torch.add(x, torch.ones_like(x))
        x = self.lp_normalization(x, p=self.p, dim=self.dim)
        x = torch.div(x, torch.ones_like(x))
        return x


class TestFuseLpNormalization(TestTransformer):
    def make_unit_model(self, input_shapes, p, dim):
        orig_model, trans_model = self.make_test_model(
            UnitTestModel(p, dim), FuseLpNormalization(), input_shapes
        )
        return orig_model, trans_model

    def make_multi_model(self, input_shapes, p, dim):
        orig_model, trans_model = self.make_test_model(
            MultiTestModel(p, dim), FuseLpNormalization(), input_shapes
        )
        return orig_model, trans_model

    def test_case1(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only LpNormalization operator
        """
        input_shapes = [(1, 4, 8)]
        p = 2
        axis = 1
        op_types = ['LpNormalization']

        orig_model, trans_model = self.make_unit_model(input_shapes, p, axis)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute(axis, trans_model.graph.node[0].attribute[0].i)
        self.check_attribute(p, trans_model.graph.node[0].attribute[1].i)

    @unittest.skipIf(
        onnxruntime.get_device() == "GPU",
        "https://github.com/furiosa-ai/furiosa-sdk-private/issues/34#issuecomment-900467342",
    )
    def test_case2(self):
        """
        Test whether the original model is well transformed for multi operator model,
         which contains operators other than LpNormalization
        """
        input_shapes = [(1, 4, 8)]
        p = 1
        axis = 0

        orig_model, trans_model = self.make_multi_model(input_shapes, p, axis)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
