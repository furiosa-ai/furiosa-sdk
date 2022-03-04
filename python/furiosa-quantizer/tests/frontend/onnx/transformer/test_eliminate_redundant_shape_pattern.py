import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.eliminate_redundant_shape_pattern import (
    EliminateRedundantShapePattern,
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
    Pattern_5,
    Pattern_6,
    Pattern_7,
    Pattern_8,
)
from tests.frontend.onnx.transformer import TestTransformer


class TestEliminateRedundantShapePattern(TestTransformer):
    def test_case1(self):
        input_shape = [1, 1, 576]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "2": (np.float32, input_shape),
            },
            "node": [
                ("Flatten", ["x"], ["0"], {"axis": 1}),
                ("Unsqueeze", ["0"], ["1"], {"axes": [1]}),
                ("Add", ["1", "2"], ["y"]),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case1_1(self):
        input_shape = [1, 1, 576]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "0": (np.float32, input_shape),
            },
            "node": [
                ("Add", ["x", "0"], ["1"]),
                ("Flatten", ["1"], ["2"], {"axis": 1}),
                ("Unsqueeze", ["2"], ["y"], {"axes": [1]}),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2(self):
        input_shape = [1, 1, 576]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape), "y1": (np.float32, output_shape)},
            "initializer": {
                "0": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
            },
            "node": [
                ("Add", ["x", "0"], ["1"]),
                ("Reshape", ["1", "reshape"], ["2"]),
                ("Flatten", ["2"], ["3"], {"axis": 1}),
                ("Unsqueeze", ["3"], ["y"], {"axes": [1]}),
                ("Reshape", ["1", "reshape"], ["4"]),
                ("Flatten", ["4"], ["5"], {"axis": 1}),
                ("Unsqueeze", ["5"], ["y1"], {"axes": [1]}),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=["Add"])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case2_1(self):
        input_shape = [1, 1, 576]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape), "y1": (np.float32, output_shape)},
            "initializer": {
                "0": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
            },
            "node": [
                ("Div", ["x", "0"], ["1"]),
                ("Reshape", ["1", "reshape"], ["2"]),
                ("Flatten", ["2"], ["3"], {"axis": 1}),
                ("Unsqueeze", ["3"], ["y"], {"axes": [1]}),
                ("Reshape", ["1", "reshape"], ["4"]),
                ("Flatten", ["4"], ["5"], {"axis": 1}),
                ("Unsqueeze", ["5"], ["6"], {"axes": [1]}),
                ("Add", ["6", "0"], ["y1"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_2)
        self.check_graph_node(trans_model, op_types=["Div", "Add"])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3(self):
        input_shape = [3, 9, 9]
        output_shape = [9, 9, 3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "1": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
                "reshape1": np.array([*input_shape[1:], input_shape[0]], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "reshape"], ["0"]),
                ("Div", ["0", "1"], ["2"]),
                ("Reshape", ["2", "reshape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Div', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case3_1(self):
        input_shape = [3, 9, 9]
        output_shape = [9, 9, 3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "1": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
                "reshape1": np.array([*input_shape[1:], input_shape[0]], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "reshape"], ["0"]),
                ("Div", ["0", "1"], ["2"]),
                ("Reshape", ["2", "reshape1"], ["3"]),
                ("Reshape", ["3", "reshape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_3)
        self.check_graph_node(trans_model, op_types=['Div', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case4(self):
        input_shape = [2, 3, 3]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "4": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
                "reshape1": np.array([np.prod(input_shape)], dtype=np.int64),
                "expand": np.array([1, np.prod(input_shape)], dtype=np.int64),
                "expand1": np.array([1, 1, np.prod(input_shape)], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "reshape1"], ["0"]),
                ("Expand", ["0", "expand"], ["1"]),
                ("Expand", ["1", "expand1"], ["2"]),
                ("Reshape", ["2", "reshape"], ["3"]),
                ("Add", ["3", "4"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_4)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case5(self):
        input_shape = [2, 3, 3]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "3": (np.float32, input_shape),
                "reshape": np.array(input_shape, dtype=np.int64),
                "reshape1": np.array([np.prod(input_shape)], dtype=np.int64),
                "expand": np.array([1, np.prod(input_shape)], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "reshape1"], ["0"]),
                ("Expand", ["0", "expand"], ["1"]),
                ("Reshape", ["1", "reshape"], ["2"]),
                ("Add", ["2", "3"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_5)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case5_1(self):
        input_shape = [2, 3, 3]
        output_shape = [3, 2, 3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "reshape": np.array([input_shape[-1], *input_shape[:-1]], dtype=np.int64),
                "reshape1": np.array([np.prod(input_shape)], dtype=np.int64),
                "expand": np.array([1, np.prod(input_shape)], dtype=np.int64),
            },
            "node": [
                ("Reshape", ["x", "reshape1"], ["0"]),
                ("Expand", ["0", "expand"], ["1"]),
                ("Reshape", ["1", "reshape"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_5)
        self.check_graph_node(trans_model, op_types=['Reshape', 'Expand', 'Reshape'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case6(self):
        input_shape = [2, 3, 3]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "0": (np.float32, input_shape),
                "reshape": np.array([input_shape[0], 1, np.prod(input_shape[1:])], dtype=np.int64),
                "reshape1": np.array(input_shape, dtype=np.int64),
            },
            "node": [
                ("Add", ["x", "0"], ["1"]),
                ("Reshape", ["1", "reshape"], ["2"]),
                ("Reshape", ["2", "reshape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_6)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case7(self):
        input_shape = [2, 3, 3]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "0": (np.float32, input_shape),
                "reshape": np.array([input_shape[0], 1, np.prod(input_shape[1:])], dtype=np.int64),
                "reshape1": np.array(input_shape, dtype=np.int64),
                "reshape2": np.array([input_shape[0], 1, *input_shape[1:]], dtype=np.int64),
            },
            "node": [
                ("Add", ["x", "0"], ["1"]),
                ("Reshape", ["1", "reshape"], ["2"]),
                ("Reshape", ["2", "reshape2"], ["3"]),
                ("Reshape", ["3", "reshape1"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_7)
        self.check_graph_node(trans_model, op_types=['Add'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case8(self):
        input_shape = [1, 16, 24, 8]
        output_shapes = [[1, 1, 3072], [3072, 1, 1], [1, 3072, 1]]

        prod = np.prod(input_shape, dtype=np.int64)
        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {
                "y": (np.float32, output_shapes[0]),
                "y1": (np.float32, output_shapes[1]),
                "y2": (np.float32, output_shapes[2]),
                "y3": (np.float32, output_shapes[2]),
            },
            "initializer": {
                "a": (np.float32, [1, 1, prod]),
                "a1": (np.float32, [1, prod, 1]),
                "shape": np.array([1, 1, prod], dtype=np.int64),
                "shape1": np.array([prod], dtype=np.int64),
                "shape2": np.array([1, prod], dtype=np.int64),
                "shape3": np.array([prod, 1, 1], dtype=np.int64),
                "shape4": np.array([1, prod, 1], dtype=np.int64),
                "shape5": np.array([1, 1, prod, 1], dtype=np.int64),
            },
            "node": [
                ("Flatten", ["x"], ["0"], {"axis": 1}),
                ("Unsqueeze", ["0"], ["1"], {"axes": [1]}),
                ("Add", ["1", "a"], ["2"]),
                ("Add", ["2", "a"], ["3"]),
                ("Flatten", ["3"], ["4"], {"axis": 1}),
                ("Unsqueeze", ["4"], ["5"], {"axes": [1]}),
                ("Add", ["5", "a"], ["6"]),
                # branch 1
                ("Reshape", ["6", "shape"], ["7"]),
                ("Flatten", ["7"], ["8"], {"axis": 1}),
                ("Unsqueeze", ["8"], ["9"], {"axes": [1]}),
                ("Reshape", ["9", "shape1"], ["10"]),
                ("Expand", ["10", "shape2"], ["11"]),
                ("Expand", ["11", "shape"], ["12"]),
                ("Reshape", ["12", "shape"], ["13"]),
                ("Add", ["13", "a"], ["14"]),
                ("Reshape", ["14", "shape1"], ["15"]),
                ("Expand", ["15", "shape2"], ["16"]),
                ("Reshape", ["16", "shape"], ["17"]),
                ("Add", ["17", "a"], ["y"]),
                # branch 2
                ("Reshape", ["6", "shape"], ["18"]),
                ("Flatten", ["18"], ["19"], {"axis": 1}),
                ("Unsqueeze", ["19"], ["20"], {"axes": [1]}),
                ("Div", ["20", "a"], ["21"]),
                # branch 2-1
                ("Reshape", ["21", "shape"], ["22"]),
                ("Flatten", ["22"], ["23"], {"axis": 1}),
                ("Unsqueeze", ["23"], ["24"], {"axes": [1]}),
                ("Reshape", ["24", "shape1"], ["25"]),
                ("Expand", ["25", "shape2"], ["26"]),
                ("Reshape", ["26", "shape3"], ["27"]),
                ("Reshape", ["27", "shape3"], ["y1"]),
                # branch 2-2
                ("Reshape", ["21", "shape"], ["28"]),
                ("Flatten", ["28"], ["29"], {"axis": 1}),
                ("Unsqueeze", ["29"], ["30"], {"axes": [1]}),
                ("Add", ["30", "a"], ["31"]),
                ("Reshape", ["31", "shape"], ["32"]),
                ("Div", ["32", "a"], ["33"]),
                ("Reshape", ["33", "shape4"], ["y2"]),
                ("Add", ["y2", "a1"], ["34"]),
                ("Reshape", ["34", "shape"], ["35"]),
                ("Reshape", ["35", "shape5"], ["36"]),
                ("Reshape", ["36", "shape4"], ["y3"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, EliminateRedundantShapePattern())
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)

    def test_case9(self):
        in_channel = 4
        input_shape = [1, in_channel, 8, 8]
        out_channel = in_channel
        output_shape = [1, out_channel, 6, 6]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 2, 2]),
                "b": (np.float32, [out_channel]),
                "expand": np.array([1, out_channel, 7, 7], dtype=np.int64),
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Expand", ["0", "expand"], ["1"]),
                ("Conv", ["1", "w", "b"], "y"),
            ],
        }
        orig_model, trans_model = self.make_test_model(model_desc, Pattern_8)
        self.check_graph_node(trans_model, op_types=['Conv', 'Conv'])
        self.check_output_value(orig_model, trans_model, [input_shape])
        self.check_value_info(trans_model)
