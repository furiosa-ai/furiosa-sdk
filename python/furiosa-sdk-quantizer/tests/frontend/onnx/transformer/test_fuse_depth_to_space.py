import abc

import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_depth_to_space import FuseDepthToSpace
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module, abc.ABC):
    """
    DepthToSpace DRC mode
        https://github.com/onnx/onnx/blob/master/docs/Operators.md#depthtospace
        b, c, h, w = x.shape
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize]

    DepthToSpace CRD mode
        https://github.com/onnx/onnx/blob/master/docs/Operators.md#depthtospace
        b, c, h, w = x.shape
        tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
        tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
        y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
    """

    def __init__(self, input_shape, blocksize, mode):
        super(UnitTestModel, self).__init__()
        b, c, h, w = input_shape

        channel_split = c // (blocksize ** 2)
        assert channel_split * blocksize ** 2 == c

        if mode == 'DCR':
            self.permute = [0, 3, 4, 1, 5, 2]
            self.shape_0 = [b, blocksize, blocksize, channel_split, h, w]
        elif mode == 'CRD':
            self.permute = [0, 1, 4, 2, 5, 3]
            self.shape_0 = [b, channel_split, blocksize, blocksize, h, w]
        else:
            raise Exception()

        self.shape_1 = [b, channel_split, h * blocksize, w * blocksize]

    def forward(self, x):
        x = torch.reshape(x, shape=self.shape_0)
        x = x.permute(dims=self.permute)
        x = torch.reshape(x, shape=self.shape_1)
        return x


class MultiTestModel(UnitTestModel, abc.ABC):
    def __init__(self, input_shape, blocksize, mode):
        super(MultiTestModel, self).__init__(input_shape, blocksize, mode)

    def forward(self, x):
        x = torch.add(x, x)
        x = torch.reshape(x, shape=self.shape_0)
        x = x.permute(dims=self.permute)
        x = torch.reshape(x, shape=self.shape_1)
        x = torch.mul(x, torch.ones_like(x))
        return x


class TestReifyDepthToSpace(TestTransformer):
    def make_unit_model(self, input_shapes, blocksize, mode):
        orig_model, trans_model = self.make_test_model(
            UnitTestModel(input_shapes[0], blocksize, mode), FuseDepthToSpace(), input_shapes
        )
        return orig_model, trans_model

    def make_multi_model(self, input_shapes, blocksize, mode):
        orig_model, trans_model = self.make_test_model(
            UnitTestModel(input_shapes[0], blocksize, mode), FuseDepthToSpace(), input_shapes
        )
        return orig_model, trans_model

    def test_case1(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only DepthToSpace operator
        """
        input_shapes = [(1, 4, 8, 8)]
        blocksize = 2
        mode = 'DCR'
        op_types = ['DepthToSpace']

        orig_model, trans_model = self.make_unit_model(input_shapes, blocksize, mode)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute(blocksize, trans_model.graph.node[0].attribute[0].i)
        self.check_attribute(str.encode(mode), trans_model.graph.node[0].attribute[1].s)

    def test_case2(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only DepthToSpace operator
        """
        input_shapes = [(1, 9, 4, 4)]
        blocksize = 3
        mode = 'CRD'
        op_types = ['DepthToSpace']

        orig_model, trans_model = self.make_unit_model(input_shapes, blocksize, mode)
        self.check_graph_node(trans_model, op_types)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
        self.check_attribute(blocksize, trans_model.graph.node[0].attribute[0].i)
        self.check_attribute(str.encode(mode), trans_model.graph.node[0].attribute[1].s)

    def test_case3(self):
        """
        Test whether the original model is well transformed for multi operator model,
         which contains operators other than DepthToSpace
        """
        input_shapes = [(1, 4, 8, 8)]
        blocksize = 2
        mode = 'CRD'

        orig_model, trans_model = self.make_multi_model(input_shapes, blocksize, mode)
        self.check_output_value(orig_model, trans_model, input_shapes)
        self.check_value_info(trans_model)
