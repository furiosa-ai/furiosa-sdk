import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_pad import FusePad, Pattern_1, Pattern_2
from tests.frontend.onnx.transformer import TestTransformer


class TestFusePad(TestTransformer):
    def test_case1(self):
        input_shape = [1, 3, 16, 19]
        output_shape = [1, 3, 5, 6]
        pad0 = [0, 0, 2, 2, 0, 0, 1, 1]
        pad1 = [0, 0, 1, 1, 0, 0, 1, 1]
        pad_val0 = float('-inf')
        pad_val1 = 0.0
        pool_attr = {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1}

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "pad0": np.array(pad0, dtype=np.int64),
                "pad1": np.array(pad1, dtype=np.int64),
                "val0": np.array(pad_val0, dtype=np.float32),
                "val1": np.array(pad_val1, dtype=np.float32),
            },
            "node": [
                ("Pad", ["x", "pad0", "val0"], ["0"]),
                ("MaxPool", ["0"], ["1"], pool_attr),
                ("Pad", ["1", "pad1", "val1"], ["2"]),
                ("MaxPool", ["2"], ["y"], pool_attr),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['MaxPool', 'Pad', 'MaxPool'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case1_1(self):
        input_shape = [1, 3, 14, 16, 17]
        output_shape = [1, 3, 5, 6, 6]
        pad0 = [0, 0, 2, 3, 1, 0, 0, 1, 3, 2]
        pad_val0 = float('-inf')
        pool_attr = {"kernel_shape": [7, 7, 7], "strides": [3, 3, 3], "pads": [1, 1, 1, 1, 1, 1]}

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.float32, input_shape),
                "pad0": np.array(pad0, dtype=np.int64),
                "val0": np.array(pad_val0, dtype=np.float32),
            },
            "node": [
                ("Add", ["x", "a"], ["0"]),
                ("Pad", ["0", "pad0", "val0"], ["1"]),
                ("MaxPool", ["1"], ["y"], pool_attr),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Add', 'MaxPool'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [1, 3, 13, 11]
        output_shape = [1, 3, 1, 2]
        pad0 = [0, 0, 1, 1, 0, 0, 1, 1]
        pad1 = [0, 0, 2, 2, 0, 0, 2, 2]
        pad2 = [0, 0, 1, 2, 0, 0, 2, 1]
        pad_val0 = 0.0
        pad_val2 = 0.0
        pool_attr0 = {"kernel_shape": [3, 3], "strides": [3, 3]}
        pool_attr1 = {"kernel_shape": [4, 4], "strides": [3, 3]}
        pool_attr2 = {"kernel_shape": [5, 3], "strides": [3, 3], "pads": [1, 1, 1, 1]}

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "pad0": np.array(pad0, dtype=np.int64),
                "pad1": np.array(pad1, dtype=np.int64),
                "pad2": np.array(pad2, dtype=np.int64),
                "val0": np.array(pad_val0, dtype=np.float32),
                "val2": np.array(pad_val2, dtype=np.float32),
            },
            "node": [
                ("Pad", ["x", "pad0", "val0"], ["0"]),
                ("AveragePool", ["0"], ["1"], pool_attr0),
                ("Pad", ["1", "pad1"], ["2"]),
                ("AveragePool", ["2"], ["3"], pool_attr1),
                ("Pad", ["3", "pad2", "val2"], ["4"]),
                ("AveragePool", ["4"], ["y"], pool_attr2),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(
            trans_model, op_types=['AveragePool', 'AveragePool', 'Pad', 'AveragePool']
        )
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [1, 3, 112, 112]
        output_shape = [1, 3, 2, 2]
        pad0 = [0, 0, 2, 2, 0, 0, 1, 1]
        pad1 = [0, 0, 1, 1, 0, 0, 1, 1]
        pad2 = [0, 0, 1, 1, 0, 0, 1, 1]
        pad3 = [0, 0, 2, 2, 0, 0, 2, 2]
        pad4 = [0, 0, 1, 2, 0, 0, 2, 1]
        pad_val0 = float('-inf')
        pad_val1 = 0.0
        pad_val2 = 0.0
        pad_val4 = 0.0
        pool_attr0 = {"kernel_shape": [3, 3], "strides": [2, 2], "ceil_mode": 1}
        pool_attr2 = {"kernel_shape": [3, 3], "strides": [3, 3]}
        pool_attr3 = {"kernel_shape": [4, 4], "strides": [3, 3]}
        pool_attr4 = {"kernel_shape": [5, 5], "strides": [3, 3], "pads": [1, 1, 1, 1]}
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "pad0": np.array(pad0, dtype=np.int64),
                "pad1": np.array(pad1, dtype=np.int64),
                "pad2": np.array(pad2, dtype=np.int64),
                "pad3": np.array(pad3, dtype=np.int64),
                "pad4": np.array(pad4, dtype=np.int64),
                "val0": np.array(pad_val0, dtype=np.float32),
                "val1": np.array(pad_val1, dtype=np.float32),
                "val2": np.array(pad_val2, dtype=np.float32),
                "val4": np.array(pad_val4, dtype=np.float32),
            },
            "node": [
                ("Pad", ["x", "pad0", "val0"], ["0"]),
                ("MaxPool", ["0"], ["1"], pool_attr0),
                ("Pad", ["1", "pad1", "val1"], ["2"]),
                ("MaxPool", ["2"], ["3"], pool_attr0),
                ("Pad", ["3", "pad2", "val2"], ["4"]),
                ("AveragePool", ["4"], ["5"], pool_attr2),
                ("Pad", ["5", "pad3"], ["6"]),
                ("AveragePool", ["6"], ["7"], pool_attr3),
                ("Pad", ["7", "pad4", "val4"], ["8"]),
                ("AveragePool", ["8"], ["y"], pool_attr4),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, FusePad())
        self.check_graph_node(
            trans_model,
            op_types=[
                'MaxPool',
                'Pad',
                'MaxPool',
                'AveragePool',
                'AveragePool',
                'Pad',
                'AveragePool',
            ],
        )
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
