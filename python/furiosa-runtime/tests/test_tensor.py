import random
import unittest

import mnist
import numpy as np

from furiosa.runtime import session
from furiosa.runtime.tensor import DataType, TensorArray, numpy_dtype
from tests.test_base import MNIST_ONNX, NAMED_TENSORS_ONNX, SessionTester, assert_tensors_equal


class TestTensor(unittest.TestCase):
    mnist_images = mnist.train_images().reshape((60000, 1, 28, 28)).astype(np.float32)

    @classmethod
    def setUpClass(cls):
        cls.sess = SessionTester(MNIST_ONNX).session

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_dtype(self):
        tensor = self.sess.input(0)
        self.assertEqual(np.float32, numpy_dtype(tensor.dtype))

    def test_tensor_desc(self):
        tensor = self.sess.input(0)
        self.assertEqual(4, tensor.ndim)
        self.assertEqual("NCHW", tensor.format)
        self.assertEqual((1, 1, 28, 28), tensor.shape)
        self.assertEqual(DataType.FLOAT32, tensor.dtype)
        self.assertEqual(np.float32, numpy_dtype(tensor))
        self.assertEqual(784, tensor.length)
        self.assertEqual(3136, tensor.size)
        self.assertEqual(
            tensor.__repr__(),
            'TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)',
        )

    def test_tensor(self):
        sess = self.sess
        self.assertEqual(sess.input_num, 1)
        self.assertEqual(sess.output_num, 1)

        inputs: TensorArray = sess.allocate_inputs()
        inputs2: TensorArray = sess.allocate_inputs()
        self.assertEqual(1, len(inputs))
        self.assertEqual(1, len(inputs[0:1]))
        self.assertEqual(np.float32, numpy_dtype(inputs[0]))

        # Testing TensorArray.__repr__()
        inputs[0] = np.zeros(inputs[0].shape, np.float32)
        self.assertEqual(
            inputs[0].__repr__(), 'Tensor(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32)'
        )

        for i in range(0, 10):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.mnist_images[idx : idx + 1]

            inputs[0] = ndarray_value
            assert_tensors_equal(ndarray_value, inputs[0].numpy())

            # idempotent
            self.assertEqual(inputs[0], inputs[0])

            # equality of contents
            inputs2[0] = ndarray_value
            self.assertEqual(inputs[0], inputs2[0])


class TestTensorNames(unittest.TestCase):
    def test_tensor_names(self):
        with session.create(NAMED_TENSORS_ONNX) as sess:
            self.assertEqual(sess.input_num, 3)
            self.assertEqual(sess.output_num, 3)
            for idx, expected in enumerate(["input.1", "input.3", "input"]):
                actual_result = sess.input(idx).name
                self.assertEqual(
                    actual_result, expected, f"expected: {expected}, actual: {actual_result}"
                )
            for idx, expected in enumerate(["18_dequantized", "19_dequantized", "15_dequantized"]):
                actual_result = sess.output(idx).name
                self.assertEqual(
                    actual_result, expected, f"expected: {expected}, actual: {actual_result}"
                )

            self.assertEqual(
                sess.input(0).__repr__(),
                "TensorDesc(name=\"input.1\", shape=(1, 3, 8, 8), dtype=FLOAT32, format=NCHW, size=768, len=192)",
            )


if __name__ == '__main__':
    unittest.main()
