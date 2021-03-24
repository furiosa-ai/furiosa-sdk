# TODO refactor test case
# import unittest
# import abc
#
# import torch
# import torch.nn as nn
#
# from dss.frontend.onnx.utils.check_model import check_model
# from dss.frontend.onnx.transformer.extract_constant_to_initializer import ExtractConstantToInitializer
# from tests import torch_to_onnx
#
#
# class TestExtractConstantToInitializer(unittest.TestCase):
#     def test_case1(self):
#         class TestModel(nn.Module, abc.ABC):
#             def forward(self, x):
#                 x = torch.mul(x, torch.ones(4))
#                 x = torch.argmax(x, dim=1)
#                 return x
#
#         model = torch_to_onnx(TestModel(), torch.randn(1, 4))
#         model = ExtractConstantToInitializer().transform(model)
#         check_model(model)
#
#         # Test#1 check if Constant node is gone.
#         for node in model.graph.node:
#             self.assertNotEqual(node.op_type, 'Constant', msg='Constant node must be removed.')
