import unittest

import numpy as np
import onnxruntime

from furiosa.quantizer.frontend.onnx.transformer.fuse_lp_normalization import Pattern_1
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseLpNormalization(TestTransformer):
    def test_case1(self):
        """
        Test whether the original model is well transformed for unit operator model,
        which contains only LpNormalization operator
        """
        input_shape = [1, 4, 8]
        output_shape = input_shape
        axis = 1
        p = 2

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "min": np.array(9.999999960041972e-13, dtype=np.float32),
                "shape": np.array(input_shape, dtype=np.int64),
            },
            "node": [
                ("ReduceL2", ["x"], ["0"], {"axes": [axis], "keepdims": 1}),
                ("Clip", ["0", "min"], ["1"]),
                ("Expand", ["1", "shape"], ["2"]),
                ("Div", ["x", "2"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['LpNormalization'])
        self.check_output_value(orig_model, trans_model, [input_shape])
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
        input_shape = [1, 4, 8]
        output_shape = input_shape
        axis = 0
        p = 1

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "min": np.array(9.999999960041972e-13, dtype=np.float32),
                "shape": np.array(input_shape, dtype=np.int64),
                "a": (np.float32, input_shape),
            },
            "node": [
                ("Add", ["x", "a"], ["0"]),
                ("ReduceL1", ["0"], ["1"], {"axes": [axis], "keepdims": 1}),
                ("Clip", ["1", "min"], ["2"]),
                ("Expand", ["2", "shape"], ["3"]),
                ("Div", ["0", "3"], ["4"]),
                ("Div", ["4", "a"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Add', 'LpNormalization', 'Div'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
        self.check_attribute(axis, trans_model.graph.node[1].attribute[0].i)
        self.check_attribute(p, trans_model.graph.node[1].attribute[1].i)

    # TODO Make test case for Pattern_2.
    # See furiosa.quantizer.frontend.onnx.transformer.fuse_lp_normalization for the detailed description.
