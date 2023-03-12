import itertools
from typing import Dict, Iterable, List, Sequence, Tuple, Type, Union
import unittest

import numpy as np
import onnx
import onnxruntime as ort

from furiosa.optimizer.frontend.onnx.transformer import ONNXTransformer, utils
from furiosa.optimizer.interfaces.transformer import Transformer
from tests.frontend.onnx import make_onnx_model_from_model_desc as make_onnx_model


class TestTransformer(unittest.TestCase):
    @staticmethod
    def make_test_model(
        model_desc: Dict,
        transformer: Union[Transformer, Type[ONNXTransformer]],
    ):
        orig_model = make_onnx_model(model_desc)
        copy_model = onnx.ModelProto()  # pylint: disable=no-member
        copy_model.CopyFrom(orig_model)

        if isinstance(transformer, Transformer):
            trans_model = transformer.transform(copy_model)
        elif issubclass(transformer, ONNXTransformer):
            trans_model = transformer(copy_model).transform()
        else:
            raise TypeError(repr(transformer))

        return orig_model, trans_model

    def check_graph_node(
        self, model: onnx.ModelProto, op_types: List[str]  # pylint: disable=no-member
    ):
        assert len(model.graph.node) == len(op_types)
        for node, op_type in zip(model.graph.node, op_types):
            self.assertTrue(node.op_type == op_type)

    def check_output_value(
        self,
        orig_model: onnx.ModelProto,  # pylint: disable=no-member
        trans_model: onnx.ModelProto,  # pylint: disable=no-member
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

    @staticmethod
    def check_value_info(model):
        utils.check_value_info(model)

    def check_attribute(self, actual, expected):
        self.assertEqual(actual, expected)

    def assertListAlmostEqual(self, list1, list2, tol=2, msg=None):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol, msg=msg)

    @staticmethod
    def run_onnx_model(
        model: onnx.ModelProto, input_arrays: List[np.ndarray]  # pylint: disable=no-member
    ) -> List[np.ndarray]:
        sess = ort.InferenceSession(model.SerializeToString())
        input_names = [inp.name for inp in sess.get_inputs()]
        output_names = [out.name for out in sess.get_outputs()]
        feed_dict = dict(zip(input_names, input_arrays))
        outputs = sess.run(output_names, input_feed=feed_dict)

        return outputs

    def run_onnx_model_flatten_output(
        self, model: onnx.ModelProto, input_arrays: List[np.ndarray]  # pylint: disable=no-member
    ) -> List[List[float]]:
        outputs = self.run_onnx_model(model, input_arrays)
        return [val.flatten().tolist() for val in outputs]
