from typing import Dict, List, Type, Union
import unittest

import numpy as np
import onnx
import onnxruntime as ort

from furiosa.quantizer.frontend.onnx.transformer import ONNXTransformer
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

    def check_graph_node(self, model, op_types: List[str]):
        assert len(model.graph.node) == len(op_types)
        for node, op_type in zip(model.graph.node, op_types):
            self.assertTrue(node.op_type == op_type)

    def check_output_value(self, orig_model, trans_model, input_shapes, data=None):
        if data is None:
            rng = np.random.default_rng()
            data = [rng.standard_normal(shape, dtype=np.float32) for shape in input_shapes]

        actual = run_onnx_model(orig_model, data)
        expected = run_onnx_model(trans_model, data)

        for act, exp in zip(actual, expected):
            self.assertListAlmostEqual(act, exp, msg=f"{data}")

    def check_value_info(self, model):
        value_info = {
            vi.name: vi
            for vi in list(model.graph.value_info)
            + list(model.graph.input)
            + list(model.graph.output)
        }

        for node in model.graph.node:
            for input in list(node.input) + list(node.output):
                if not input:
                    continue
                self.assertTrue(input in value_info.keys())

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


def run_onnx_model(model: onnx.ModelProto, input_arrays: List[np.ndarray]) -> List[List[float]]:
    sess = ort.InferenceSession(model.SerializeToString())
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    feed_dict = dict(zip(input_names, input_arrays))
    outputs = sess.run(output_names, input_feed=feed_dict)

    flattened_outputs = [val.flatten().tolist() for val in outputs]

    return flattened_outputs
