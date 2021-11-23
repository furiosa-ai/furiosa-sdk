from typing import List, Sequence

import numpy as np
import onnx

from furiosa.quantizer.frontend.onnx import __OPSET_VERSION__, calibrate, quantizer
from furiosa.quantizer.frontend.onnx.quantizer.quantizer_mode import ONNXRuntimeExecutable
from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
from tests.frontend.onnx.transformer import TestTransformer, make_onnx_model


# TODO add test casess
class TestQuantOperator(TestTransformer):
    pass


# TODO add test casess
class TestDFGImportable(TestTransformer):
    pass


class TestONNXRuntimeExecutable(TestTransformer):
    def test_case1(self):
        input_shape = [1, 4, 3, 3]
        output_shape = [1, 2, 3, 3]
        input_shape = [1, 5, 2, 2]
        output_shape = [1, 6, 2, 2]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        # this test is to see if ONNXRuntimeExecutable mode works with
        # given configuration (per_channel, weight_qtype, activation_qtype),
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))
        # TODO research topic:
        # set "proper" gap between original and "well" fake-quantized model's output values
        # and use the gap to check the wellness.

    def test_case2(self):
        input_shape = [1, 5, 2, 2]
        output_shape = [1, 6, 2, 2]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))
        self.check_opset_version(quant_model, 13)

    def test_case2_a(self):
        input_shape = [1, 5, 2, 2]
        output_shape = [1, 6, 2, 2]
        opset = 12
        orig_model = _make_conv(input_shape, output_shape, opset_version=opset)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))
        self.check_opset_version(quant_model, opset)

    def test_case3(self):
        input_shape = [2, 3, 3, 3]
        output_shape = [2, 3, 3, 3]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = False
        weight_qtype = onnx.TensorProto.UINT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case4(self):
        input_shape = [1, 10, 2, 2]
        output_shape = [1, 8, 2, 2]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = True
        weight_qtype = onnx.TensorProto.UINT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        # This should fail because conv filters are asymmetrically quantized when weight_qtype=UINT is given.
        # https://github.com/furiosa-ai/furiosa-sdk-private/blob/715867969e9184ab440b87d5daeef4a22b95fc46/python/furiosa-quantizer/furiosa/quantizer/frontend/onnx/quantizer/utils.py#L179-L181
        # So the per-channel zero-points DO NOT always have the same values.
        # Meanwhile, onnxruntime requires per-channel zero-points to be identical.
        # For details, refer to the error below:
        # onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT \
        # : Non-zero status code returned while running QLinearConv node. \
        # Name:'Conv_4' Status Message: QLinearConv : filter zero point must be constant
        self.assertFalse(
            self.is_onnxruntime_executable(quant_model, [input_shape]),
        )

    def test_case5(self):
        input_shape = [3, 2, 5, 5]
        output_shape = [3, 4, 5, 5]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case6(self):
        input_shape = [1, 3, 5, 5]
        output_shape = [1, 3, 5, 5]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = True
        weight_qtype = onnx.TensorProto.UINT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        # Same goes for the comment in test_case4
        self.assertFalse(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case7(self):
        input_shape = [3, 5, 7, 7]
        output_shape = [3, 2, 7, 7]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = False
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case8(self):
        input_shape = [1, 3, 4, 4]
        output_shape = [1, 5, 4, 4]
        orig_model = _make_conv(input_shape, output_shape)

        per_channel = False
        weight_qtype = onnx.TensorProto.UINT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case9(self):
        input_shape = [1, 3, 3, 3]
        output_shape = [1, 2, 3, 3]
        orig_model = _make_conv(input_shape, output_shape, use_bias=True)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case10(self):
        input_shape = [1, 3, 3, 3]
        output_shape = [1, 2, 3, 3]
        orig_model = _make_conv(input_shape, output_shape, use_bias=True)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'Conv',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case11(self):
        input_shape = [1, 3, 5, 6]
        output_shape = [1, 4, 5, 6]
        orig_model = _make_convtranspose(input_shape, output_shape, use_bias=True)

        per_channel = True
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.UINT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'ConvTranspose',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def test_case12(self):
        input_shape = [1, 7, 9, 9]
        output_shape = [1, 14, 9, 9]
        orig_model = _make_convtranspose(input_shape, output_shape, use_bias=True)

        per_channel = False
        weight_qtype = onnx.TensorProto.INT8
        activation_qtype = onnx.TensorProto.INT8
        quant_model = _make_fake_quant_model(
            orig_model, per_channel, weight_qtype, activation_qtype
        )

        self.check_graph_node(
            quant_model,
            op_types=[
                'QuantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'DequantizeLinear',
                'ConvTranspose',
                'QuantizeLinear',
                'DequantizeLinear',
            ],
        )
        self.assertTrue(self.is_onnxruntime_executable(quant_model, [input_shape]))

    def check_opset_version(self, model: onnx.ModelProto, opset: int) -> None:
        self.assertEqual(model.opset_import[0].version, opset)

    def is_onnxruntime_executable(
        self, model: onnx.ModelProto, input_shapes: List[Sequence[int]]
    ) -> bool:
        rng = np.random.default_rng()
        data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        try:
            self.run_onnx_model(model, data)
            return True
        except Exception as onnxruntime_error:
            if all(
                msg in str(onnxruntime_error)
                for msg in [
                    "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : "
                    "Non-zero status code returned while running QLinearConv node.",
                    "QLinearConv : zero point of per-channel filter must be same",
                ]
            ):
                return False
            raise onnxruntime_error


def _make_fake_quant_model(
    model: onnx.ModelProto,
    per_channel: bool,
    weight_qtype: int,
    activation_qtype: int,
) -> onnx.ModelProto:
    dynamic_ranges = calibrate.calibrate_with_random_data(model, dataset_size=1)
    onnx_quantizer = quantizer.FuriosaONNXQuantizer(
        model=model,
        per_channel=per_channel,
        static=True,
        mode=QuantizationMode.FAKE,
        dynamic_ranges=dynamic_ranges,
    )
    onnx_quantizer.weight_qtype = weight_qtype
    onnx_quantizer.activation_qtype = activation_qtype
    onnx_quantizer.quantize_model()
    onnx_quantizer.model = onnx_quantizer.make_intermediate_representation()
    onnx_quantizer.model = ONNXRuntimeExecutable(onnx_quantizer.model, raw_data=True).transform()
    onnx_quantizer.check_model()
    return onnx_quantizer.model


def _make_conv(
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    use_bias: bool = False,
    opset_version: int = __OPSET_VERSION__,
) -> onnx.ModelProto:
    in_channel = input_shape[1]
    out_channel = output_shape[1]

    model_desc = {
        "input": {"x": (np.float32, input_shape)},
        "output": {"y": (np.float32, output_shape)},
        "initializer": {
            "w": (np.float32, [out_channel, in_channel, 1, 1]),
        },
        "node": [("Conv", ["x", "w"], ["y"])],
    }
    assert opset_version in [
        12,
        13,
    ], f"Unsupported opset_version: {opset_version}. opset_version should be in [12, 13]"
    if opset_version != __OPSET_VERSION__:
        model_desc.update({"opsetid": [("", opset_version)]})

    if use_bias:
        model_desc["initializer"].update({"b": (np.float32, [out_channel])})
        model_desc["node"] = [("Conv", ["x", "w", "b"], ["y"])]

    return make_onnx_model(model_desc)


def _make_convtranspose(
    input_shape: Sequence[int], output_shape: Sequence[int], use_bias: bool = False
) -> onnx.ModelProto:
    in_channel = input_shape[1]
    out_channel = output_shape[1]

    model_desc = {
        "input": {"x": (np.float32, input_shape)},
        "output": {"y": (np.float32, output_shape)},
        "initializer": {
            "w": (np.float32, [in_channel, out_channel, 1, 1]),
        },
        "node": [("ConvTranspose", ["x", "w"], ["y"])],
    }

    if use_bias:
        model_desc["initializer"].update({"b": (np.float32, [out_channel])})
        model_desc["node"] = [("ConvTranspose", ["x", "w", "b"], ["y"])]

    return make_onnx_model(model_desc)
