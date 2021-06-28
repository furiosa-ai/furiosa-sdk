import abc

import torch
import torch.nn as nn

from furiosa_sdk_quantizer.frontend.onnx.transformer.eliminate_argmax_output import EliminateArgmaxOutput
from tests.frontend.onnx.transformer import TestTransformer


class MultiTestModel(nn.Module, abc.ABC):
    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = torch.argmax(x, dim=2)
        return x


class MultiTestModel1(nn.Module, abc.ABC):
    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        x = torch.argmax(x, dim=1)
        x = torch.sub(x, torch.ones(x.shape))
        return x


class MultiTestModel2(nn.Module, abc.ABC):
    def forward(self, x):
        x = torch.mul(x, torch.ones(x.shape))
        y = torch.argmax(x, dim=0)
        z = torch.argmax(x, dim=0)

        return y, z


class TestEliminateArgmaxOutput(TestTransformer):
    def make_multi_model(self, torch_model, input_shapes):
        orig_model, trans_model = self.make_test_model(torch_model, EliminateArgmaxOutput(), input_shapes)
        return orig_model, trans_model

    def test_case1(self):
        """
            Test whether the original model is well transformed for unit operator model,
            which contains only Identity operator
        """
        input_shapes = [(1, 1, 4, 4)]
        op_types = ['Mul']

        orig_model, trans_model = self.make_multi_model(MultiTestModel(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_assertion(self.check_output_value,
                             {'orig_model': orig_model, 'trans_model': trans_model, 'input_shapes': input_shapes})
        self.check_value_info(trans_model)

    def test_case2(self):
        """
            Test whether the original model is well transformed for multi operator model,
             which contains operators other than Identity
        """
        input_shapes = [(2, 4, 8, 8, 16)]
        op_types = ['Mul', 'ArgMax', 'Cast', 'Sub']
        orig_model, trans_model = self.make_multi_model(MultiTestModel1(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)

    def test_case3(self):
        """
            Test whether the original model is well transformed for multi operator model,
             which contains operators other than Identity
        """
        input_shapes = [(8, 8, 8)]
        op_types = ['Mul']
        orig_model, trans_model = self.make_multi_model(MultiTestModel2(), input_shapes)
        self.check_graph_node(trans_model, op_types)
        self.check_assertion(self.check_output_value,
                             {'orig_model': orig_model, 'trans_model': trans_model, 'input_shapes': input_shapes})
        self.check_value_info(trans_model)
