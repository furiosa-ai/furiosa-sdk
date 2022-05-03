import itertools
from typing import Dict, Iterable, List, Sequence, Tuple, Type, Union
import unittest

import numpy as np
import onnx
import onnxruntime as ort

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer, utils
from furiosa.quantizer.interfaces.transformer import Transformer
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model


class TestTransformer(unittest.TestCase):
    @staticmethod
    def make_test_model(
        model_desc: Dict,
        transformer: Union[Transformer, Type[ONNXTransformer]],
    ):
        orig_model = make_onnx_model(model_desc)
        copy_model = onnx.ModelProto()
        copy_model.CopyFrom(orig_model)

        if isinstance(transformer, Transformer):
            trans_model = transformer.transform(copy_model)
        elif issubclass(transformer, ONNXTransformer):
            trans_model = transformer(copy_model).transform()
        else:
            raise TypeError(repr(transformer))

        return orig_model, trans_model

    @staticmethod
    def make_test_unit_model_from_onnx(onnx_model: onnx.ModelProto, transformer: Transformer):
        copy_model = onnx.ModelProto()

        copy_model.CopyFrom(onnx_model)
        trans_model = transformer.transform(copy_model)

        return onnx_model, trans_model

    def check_graph_node(self, model: onnx.ModelProto, op_types: List[str]):
        assert len(model.graph.node) == len(op_types)
        for node, op_type in zip(model.graph.node, op_types):
            self.assertTrue(node.op_type == op_type)

    def check_output_value(
        self,
        orig_model: onnx.ModelProto,
        trans_model: onnx.ModelProto,
        input_shapes: Iterable[Sequence[int]],
        data: List[np.ndarray] = None,
    ):
        if data is None:
            rng = np.random.default_rng()
            data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        actual = self.run_onnx_model_flatten_output(orig_model, data)
        expected = self.run_onnx_model_flatten_output(trans_model, data)

        for act, exp in zip(actual, expected):
            self.assertListAlmostEqual(act, exp, msg=f"{data}")

    def check_original_and_fake_output_value(
        self,
        orig_model: onnx.ModelProto,
        fake_quant_model: onnx.ModelProto,
        input_shapes: Iterable[Sequence[int]],
        data: List[np.ndarray] = None,
    ):
        # this function can be used in two cases:
        # Case 1. orig_model and fake_quant_model are fake quant model generated from the same model, except for the optimization.
        # for example, orig_model : model -> fake quantize, fake_quant_model : model -> eliminate clipper -> fake quantize
        # Case 2. fake_quant_model is fake quantized model of orig_model.
        fake_model_graph = fake_quant_model.graph
        fake_model_producer_map = {node.output[0]: node for node in fake_model_graph.node}

        if data is None:
            rng = np.random.default_rng()
            data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        orig_output = self.run_onnx_model_output_dict(orig_model, data)
        fake_quant_output = self.run_onnx_model_output_dict(fake_quant_model, data)

        self.AssertFakeAlmostEqual(
            fake_model_graph, orig_output, fake_quant_output, fake_model_producer_map
        )

    def AssertFakeAlmostEqual(
        self,
        fake_model_graph: onnx.GraphProto,
        orig_output: Dict[str, np.ndarray],
        fake_quant_output: Dict[str, np.ndarray],
        fake_model_producer_map: Dict[str, onnx.NodeProto],
    ):
        def _fake_quantize_data(data: np.ndarray, scale: np.ndarray, zp: np.ndarray) -> np.ndarray:
            assert scale.size == 1 and zp.size == 1
            quantized_data = np.round(data / scale) + zp
            data_dtype = data.dtype
            quant_dtype = zp.dtype
            np_dtype_info = np.iinfo(zp.dtype)
            quantized_data = np.clip(quantized_data, np_dtype_info.min, np_dtype_info.max).astype(
                quant_dtype
            )

            fake_quantized_data = (
                quantized_data.astype(data_dtype) - zp.astype(data_dtype)
            ) * scale.astype(data_dtype)
            return fake_quantized_data

        for (original_output_name, dequantized_output_name) in zip(orig_output, fake_quant_output):
            assert dequantized_output_name.endswith(
                '_dequantized'
            ), f"fake_quant output name({dequantized_output_name}) should end with _dequantized suffix. Is {fake_model_graph.name} really a fake quantized model?"
            assert original_output_name.replace(
                '_dequantized', ''
            ) == dequantized_output_name.replace(
                '_dequantized', ''
            ), f"original output name({original_output_name}) and fake_quant output name({dequantized_output_name}) should be same except for _dequantized suffix"

            scale, zp = self._get_quant_param(
                fake_model_graph, fake_model_producer_map[dequantized_output_name]
            )

            original_fake_quantized_output = _fake_quantize_data(
                orig_output[original_output_name], scale, zp
            )
            fake_model_output = fake_quant_output[dequantized_output_name]
            max_diff = np.max(np.abs(original_fake_quantized_output - fake_model_output))
            self.assertLessEqual(
                round(max_diff / scale), 1
            )  # check if max i8 quantized error is less than or equal to 1

    @staticmethod
    def check_value_info(model):
        utils.check_value_info(model)

    def check_value_info_with_warning(self, model, num_warning):
        # check value info and assert number of warning for value_info with elem_type != FLOAT
        with self.assertLogs("Furiosa-Quantizer", level="WARNING") as cm:
            self.check_value_info(model)
            self.assertEqual(len(cm.output), num_warning)

    def check_initializer(self, actual, expected):
        self.assertListAlmostEqual(actual.flatten().tolist(), expected.flatten().tolist())

    def check_attribute(self, actual, expected):
        self.assertEqual(actual, expected)

    def check_assertion(self, func, kwargs):
        self.assertRaises(AssertionError, func, **kwargs)

    def assertListAlmostEqual(self, list1, list2, tol=2, msg=None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol, msg=msg)

    @staticmethod
    def run_onnx_model(model: onnx.ModelProto, input_arrays: List[np.ndarray]) -> List[np.ndarray]:
        sess = ort.InferenceSession(model.SerializeToString())
        input_names = [inp.name for inp in sess.get_inputs()]
        output_names = [out.name for out in sess.get_outputs()]
        feed_dict = dict(zip(input_names, input_arrays))
        outputs = sess.run(output_names, input_feed=feed_dict)

        return outputs

    def run_onnx_model_flatten_output(
        self, model: onnx.ModelProto, input_arrays: List[np.ndarray]
    ) -> List[List[float]]:
        outputs = self.run_onnx_model(model, input_arrays)
        return [val.flatten().tolist() for val in outputs]

    def run_onnx_model_output_dict(
        self, model: onnx.ModelProto, input_arrays: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        outputs = self.run_onnx_model(model, input_arrays)
        output_names = [out.name for out in model.graph.output]
        return dict(zip(output_names, outputs))

    @staticmethod
    def _get_quant_param(graph: onnx.GraphProto, node: onnx.NodeProto) -> Tuple[np.ndarray]:
        assert node.op_type in ['QuantizeLinear', 'DequantizeLinear']
        quant_param_dict = {
            tensor.name: onnx.numpy_helper.to_array(tensor)
            for tensor in graph.initializer
            if tensor.name in [node.input[1], node.input[2]]
        }
        return (quant_param_dict[node.input[1]], quant_param_dict[node.input[2]])
