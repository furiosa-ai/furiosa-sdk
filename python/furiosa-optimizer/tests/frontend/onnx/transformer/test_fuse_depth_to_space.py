import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_depth_to_space import Pattern_1
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseDepthToSpace(TestTransformer):
    def test_case1(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only DepthToSpace operator
        """
        input_shape = [1, 4, 8, 8]
        output_shape = [1, 1, 16, 16]
        blocksize = 2
        mode = 'DCR'
        pre_reshape, permute, post_reshape = _get_args(input_shape, blocksize, mode)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"shape": np.array(pre_reshape), "shape1": np.array(post_reshape)},
            "node": [
                ("Reshape", ["x", "shape"], ["0"]),
                ("Transpose", ["0"], ["1"], {"perm": permute}),
                ("Reshape", ["1", "shape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['DepthToSpace'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
        self.check_attribute(blocksize, trans_model.graph.node[0].attribute[0].i)
        self.check_attribute(str.encode(mode), trans_model.graph.node[0].attribute[1].s)

    def test_case2(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only DepthToSpace operator
        """
        input_shape = [1, 9, 4, 4]
        output_shape = [1, 1, 12, 12]
        blocksize = 3
        mode = 'CRD'
        pre_reshape, permute, post_reshape = _get_args(input_shape, blocksize, mode)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {"shape": np.array(pre_reshape), "shape1": np.array(post_reshape)},
            "node": [
                ("Reshape", ["x", "shape"], ["0"]),
                ("Transpose", ["0"], ["1"], {"perm": permute}),
                ("Reshape", ["1", "shape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['DepthToSpace'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
        self.check_attribute(blocksize, trans_model.graph.node[0].attribute[0].i)
        self.check_attribute(str.encode(mode), trans_model.graph.node[0].attribute[1].s)

    def test_case3(self):
        """
        Test whether the original model is well transformed for multi operator model,
         which contains operators other than DepthToSpace
        """
        input_shape = [1, 4, 8, 8]
        output_shape = [1, 1, 16, 16]
        blocksize = 2
        mode = 'CRD'
        pre_reshape, permute, post_reshape = _get_args(input_shape, blocksize, mode)

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "shape": np.array(pre_reshape),
                "shape1": np.array(post_reshape),
                "w": (np.float32, output_shape),
            },
            "node": [
                ("Add", ["x", "x"], ["0"]),
                ("Reshape", ["0", "shape"], ["1"]),
                ("Transpose", ["1"], ["2"], {"perm": permute}),
                ("Reshape", ["2", "shape1"], ["3"]),
                ("Mul", ["3", "w"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)


def _get_args(input_shape, blocksize, mode):
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

    b, c, h, w = input_shape

    channel_split = c // (blocksize**2)
    assert channel_split * blocksize**2 == c

    assert mode in ['DCR', 'CRD'], f"Unknown mode: {mode}. 'mode' must be either 'DCR' or 'CRD'."

    if mode == 'DCR':
        permute = [0, 3, 4, 1, 5, 2]
        pre_reshape = [b, blocksize, blocksize, channel_split, h, w]
    elif mode == 'CRD':
        permute = [0, 1, 4, 2, 5, 3]
        pre_reshape = [b, channel_split, blocksize, blocksize, h, w]

    post_reshape = [b, channel_split, h * blocksize, w * blocksize]

    return pre_reshape, permute, post_reshape
