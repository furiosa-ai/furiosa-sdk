from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx import calibrate, quantizer
from furiosa.quantizer.frontend.onnx.quantizer import quantizer_mode
from furiosa.quantizer.frontend.onnx.quantizer.eliminate_clipper import (
    ClipperElimination,
    Pattern_1,
    Pattern_2,
    Pattern_3,
    Pattern_4,
    Pattern_5,
    Pattern_6,
)
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from tests.frontend.onnx.transformer import TestTransformer, make_onnx_model


class TestEliminateClipper(TestTransformer):
    def check_output_value_using_FAKE_mode(
        self,
        orig_model: onnx.ModelProto,
        trans_model: onnx.ModelProto,
        input_shapes: List[Tuple],
    ) -> None:
        graph = orig_model.graph
        producer_map = {node.output[0]: node for node in graph.node}

        # Check output value by using FAKE mode to ensure given model is onnxruntime runnable
        rng = np.random.default_rng()
        data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        orig_output = self.run_onnx_model_output_dict(
            quantizer_mode.ONNXRuntimeExecutable(orig_model, raw_data=True).transform(), data
        )
        trans_output = self.run_onnx_model_output_dict(
            quantizer_mode.ONNXRuntimeExecutable(trans_model, raw_data=True).transform(), data
        )

        # For all outputs produced, max_diff / scale should be "very"(<5e-06) close to round(max_diff / scale).
        # In other words, if well transformed, the output error(s) between orig model and trans model should be only caused by
        # Q-DQ before clipper in the former, supposed to be removed in the latter.
        for k in orig_output:
            scale, _ = self._get_quant_param(graph, producer_map[k])
            max_diff = np.max(np.abs(orig_output[k] - trans_output[k]))
            self.assertAlmostEqual(max_diff / scale, round(max_diff / scale), places=4)

    def check_quant_params(
        self, qlinear: onnx.NodeProto, dqlinear: onnx.NodeProto, graph: onnx.GraphProto
    ) -> None:
        assert qlinear.op_type == "QuantizeLinear", repr(qlinear)
        assert dqlinear.op_type == "DequantizeLinear", repr(dqlinear)

        for qparam_qlinear, qparam_dqlinear in zip(
            self._get_quant_param(graph, qlinear), self._get_quant_param(graph, dqlinear)
        ):
            self.check_initializer(
                qparam_qlinear,
                qparam_dqlinear,
            )

    def test_case1(self):
        in_channel = 16
        input_shape = [2, in_channel, 4, 4]
        out_channel = 8
        output_shape = [2, out_channel, 4, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
            },
            "node": [
                ("Conv", ["x", "w"], ["0"]),
                ("Relu", ["0"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_1)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=3)

    def test_case2(self):
        in_channel = 6
        input_shape = [3, in_channel, 10, 10]
        out_channel = 9
        output_shape = [3, out_channel, 8, 8]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "min": np.array(0.0, dtype=np.float32),
                "max": np.array(6.0, dtype=np.float32),
            },
            "node": [
                ("Conv", ["x", "w"], ["0"]),
                ("Clip", ["0", "min", "max"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_2)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=3)

    def test_case3(self):
        input_shape = [1, 3, 8, 8]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.float32, input_shape),
            },
            "node": [
                ("Add", ["x", "a"], ["0"]),
                ("Relu", ["0"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_3)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=3)

    def test_case4(self):
        input_shape = [1, 3, 8, 8]
        output_shape = input_shape

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "a": (np.float32, input_shape),
                "min": np.array(-1.0, dtype=np.float32),
                "max": np.array(1.0, dtype=np.float32),
            },
            "node": [
                ("Add", ["x", "a"], ["0"]),
                ("Clip", ["0", "min", "max"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_4)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Add',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=3)

    def test_case5(self):
        in_channel = out_channel = 32
        input_shape = output_shape = [1, in_channel]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
            },
            "node": [
                ("Unsqueeze", ["x"], ["0"], {"axes": [2, 3]}),
                ("Conv", ["0", "w"], ["1"]),
                ("Squeeze", ["1"], ["2"], {"axes": [2, 3]}),
                ("Relu", ["2"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_5)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Unsqueeze',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Squeeze',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=5)

    def test_case6(self):
        in_channel = out_channel = 32
        input_shape = output_shape = [1, in_channel]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {"y": (np.float32, output_shape)},
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 1, 1]),
                "min": np.array(0.0, dtype=np.float32),
                "max": np.array(1.0, dtype=np.float32),
            },
            "node": [
                ("Unsqueeze", ["x"], ["0"], {"axes": [2, 3]}),
                ("Conv", ["0", "w"], ["1"]),
                ("Squeeze", ["1"], ["2"], {"axes": [2, 3]}),
                ("Clip", ["2", "min", "max"], ["y"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_6)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Unsqueeze',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'Squeeze',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=5)

    def test_case7(self):
        in_channel = 3
        input_shape = [1, in_channel, 4, 4]
        out_channel = 2
        output_shape = [1, out_channel, 2, 2]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {
                "y": (np.float32, output_shape),
                "y1": (np.float32, output_shape),
                "y2": (np.float32, output_shape),
            },
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
            },
            "node": [
                ("Conv", ["x", "w"], ["0"]),
                ("Relu", ["0"], ["y"]),
                ("Relu", ["0"], ["y1"]),
                ("Relu", ["0"], ["y2"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_1)
        self.check_graph_node(
            trans_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        # compare QuantizeLinear_5's quantization params with DequantizeLinear_8's
        qlinear1 = graph.node[-4]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        # compare QuantizeLinear_5's quantization params with DequantizeLinear_7's
        dqlinear1 = graph.node[-2]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        # compare QuantizeLinear_5's quantization params with DequantizeLinear_8's
        dqlinear1 = graph.node[-3]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=3)

    def test_case8(self):
        in_channel = 8
        input_shape = [1, in_channel, 6, 6]
        out_channel = 4
        output_shape = [1, out_channel, 4, 4]

        model_desc = {
            "input": {"x": (np.float32, input_shape)},
            "output": {
                "y": (np.float32, output_shape),
                "y1": (np.float32, output_shape),
            },
            "initializer": {
                "w": (np.float32, [out_channel, in_channel, 3, 3]),
                "min": np.array(0.0, dtype=np.float32),
                "max": np.array(6.0, dtype=np.float32),
            },
            "node": [
                ("Conv", ["x", "w"], ["0"]),
                ("Relu", ["0"], ["y"]),
                ("Clip", ["0", "min", "max"], ["y1"]),
            ],
        }

        onnx_model = make_onnx_model(model_desc)
        orig_model, trans_model = _make_test_model(onnx_model, Pattern_1)
        self.check_graph_node(
            trans_model,
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
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
                'Clip',
                'QuantizeLinear',
                'DequantizeLinear',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        graph = trans_model.graph
        # compare QuantizeLinear_16's quantization params with DequantizeLinear_17's
        qlinear1 = graph.node[-2]
        dqlinear1 = graph.node[-1]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        # compare QuantizeLinear_14's quantization params with DequantizeLinear_15's
        qlinear1 = graph.node[-4]
        dqlinear1 = graph.node[-3]
        self.check_quant_params(qlinear1, dqlinear1, graph)
        self.check_output_value_using_FAKE_mode(orig_model, trans_model, [input_shape])
        self.check_value_info_with_warning(trans_model, num_warning=7)


def _get_intermedidate_representation(
    model: onnx.ModelProto,
    dynamic_ranges: Dict[str, Tuple[float, float]],
) -> onnx.ModelProto:
    onnx_quantizer = quantizer.FuriosaONNXQuantizer(
        model=model,
        per_channel=True,
        static=True,
        mode=QuantizationMode.DFG,
        dynamic_ranges=dynamic_ranges,
    )
    onnx_quantizer.quantize_model()
    return onnx_quantizer.make_intermediate_representation()


def _make_test_model(
    model: onnx.ModelProto,
    pattern: ClipperElimination,
    dynamic_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
    if dynamic_ranges is None:
        dynamic_ranges = calibrate.calibrate_with_random_data(model, dataset_size=1)
    orig_model = _get_intermedidate_representation(model, dynamic_ranges)

    copy_model = onnx.ModelProto()
    copy_model.CopyFrom(orig_model)

    transformer = pattern(copy_model)
    trans_model = transformer.transform()

    return orig_model, trans_model
