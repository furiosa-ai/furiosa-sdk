from collections import defaultdict
import itertools

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx import calibrate
from furiosa.quantizer.frontend.onnx.quantizer.quantizer import (
    FuriosaONNXQuantizer,
    QuantizationMode,
)
from tests.frontend.onnx.transformer import TestTransformer, make_onnx_model


class TestFuriosaONNXQuantizer(TestTransformer):
    def test_make_intermediate_representation(self):
        in_channel = 2
        input_shape = [8, in_channel, 3, 3]
        out_channel = 4
        output_shape = [8, out_channel, 3, 3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape), "y1": (np.float32, input_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "a": (np.float32, input_shape),
            },
            "node": [("Conv", ["x", "w"], ["y"]), ("Add", ["x", "a"], ["y1"])],
        }

        onnx_model = make_onnx_model(model_desc)
        inter_repr = _make_intermediate_representation(onnx_model)
        self.check_graph_node(
            inter_repr,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Conv')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Conv').input[0],
            "x_dequantized_0",
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Add')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Add').input[0],
            "x_dequantized_1",
        )
        self._check_tensor_name(inter_repr, inter_repr.graph.output[0].name, "y_dequantized")
        self._check_tensor_name(inter_repr, inter_repr.graph.output[1].name, "y1_dequantized")

    def test_make_intermediate_representation_1(self):
        in_channel = 16
        input_shape = [2, in_channel, 4, 4]
        out_channel = 8
        output_shape = [2, out_channel, 4, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape), "y1": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
            },
            "node": [
                ("Conv", ["x", "w"], ["0"]),
                ("Relu", ["0"], ["y"]),
                ("Celu", ["0"], ["y1"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        inter_repr = _make_intermediate_representation(onnx_model)
        self.check_graph_node(
            inter_repr,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Relu',
                'DequantizeLinear',
                'Celu',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Relu')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Relu').input[0],
            "0_dequantized_0",
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Celu')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Celu').input[0],
            "0_dequantized_1",
        )
        self._check_tensor_name(inter_repr, inter_repr.graph.output[0].name, "y_dequantized")
        self._check_tensor_name(inter_repr, inter_repr.graph.output[1].name, "y1_dequantized")

    def test_make_intermediate_representation_2(self):
        input_shape = [3]
        output_shape = [3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "node": [
                ("Relu", ["x"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        inter_repr = _make_intermediate_representation(onnx_model)

        self.check_graph_node(
            inter_repr,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Relu',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Relu')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Relu').input[0],
            "x_dequantized",
        )

    def test_make_intermediate_representation_3(self):
        input_shape = [1, 8]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "node": [
                ("Relu", ["x"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        onnx_model.graph.output.append(onnx_model.graph.output[0])
        inter_repr = _make_intermediate_representation(onnx_model)
        self.check_graph_node(
            inter_repr,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Relu',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Relu')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Relu').input[0],
            "x_dequantized",
        )

    def test_make_intermediate_representation_4(self):
        in_channel = 8
        input_shape = [1, in_channel, 3, 3]
        out_channel = 8
        output_shape = [1, out_channel, 3, 3]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "b": (np.float32, [out_channel]),
            },
            "node": [
                ("Conv", ["x", "w", "b"], ["0"]),
                ("Sigmoid", ["0"], ["1"]),
                ("Add", ["1", "0"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        inter_repr = _make_intermediate_representation(onnx_model)

        self.check_graph_node(
            inter_repr,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Sigmoid',
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Conv')
        self._check_dqlinear_consumer_is_unique(inter_repr, op_type='Sigmoid')
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Sigmoid').input[0],
            "0_dequantized_0",
        )
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Add').input[0],
            "1_dequantized",
        )
        self._check_tensor_name(
            inter_repr,
            next(node for node in inter_repr.graph.node if node.op_type == 'Add').input[1],
            "0_dequantized_1",
        )
        self._check_tensor_name(inter_repr, inter_repr.graph.output[0].name, "y_dequantized")

    def _check_dqlinear_consumer_is_unique(self, model: onnx.ModelProto, op_type: str) -> None:
        consumer_map = defaultdict(list)
        for node in model.graph.node:
            for tensor in node.input:
                consumer_map[tensor].append(node)

        for node in model.graph.node:
            if node.op_type != op_type:
                continue
            for node_input in node.input:
                # DequantizeLinear output should correspond to no more than one consumer
                self.assertTrue('_dequantized' in node_input)
                self.assertEqual(len(consumer_map.get(node_input)), 1)

    def _check_tensor_name(
        self, model: onnx.ModelProto, actual_tensor: str, expected_tensor: str
    ) -> None:
        value_info_map = {
            vi.name: vi
            for vi in itertools.chain(model.graph.input, model.graph.value_info, model.graph.output)
        }
        self.assertTrue(actual_tensor in value_info_map)
        self.assertEqual(actual_tensor, expected_tensor)


def _make_intermediate_representation(model: onnx.ModelProto) -> onnx.ModelProto:
    quantizer = FuriosaONNXQuantizer(
        model,
        False,
        True,
        QuantizationMode.DFG,
        dynamic_ranges=calibrate.calibrate_with_random_data(model),
    )
    quantizer.quantize_model()
    return quantizer.make_intermediate_representation()
