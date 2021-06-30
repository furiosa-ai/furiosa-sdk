# TODO refactor test case
# import unittest
#
# import torch
# import torch.nn as nn
#
# from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model
# from furiosa_sdk_quantizer.frontend.onnx.transformer.eliminate_detection_postprocess import EliminateSSDDetectionPostprocess
# from tests import torch_to_onnx
#
#
# class TestEliminateDetectionPostprocess(unittest.TestCase):
#     def test_elimination(self):
#         class TestModel(nn.Module):
#             def __init__(self):
#                 super(TestModel, self).__init__()
#                 self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1)
#                 self.conv2 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=1)
#                 self.conv3 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=1)
#
#             def forward(self, x):
#                 x = self.conv1(x)
#                 y = self.conv2(x)
#                 z = self.conv3(x)
#                 y = torch.mul(y, torch.ones(12))
#                 z = torch.add(z, torch.ones(12))
#                 return y, z
#
#         model = torch_to_onnx(TestModel(), torch.randn(1, 3, 12, 12))
#         model = EliminateSSDDetectionPostprocess(['8', '9']).transform(model)
#         check_model(model)
#
#         self.assertEqual(len(model.graph.node), 3)
#         self.assertEqual(model.graph.node[0].name, 'Conv_0')
#         self.assertEqual(model.graph.node[1].name, 'Conv_1')
#         self.assertEqual(model.graph.node[2].name, 'Conv_2')
