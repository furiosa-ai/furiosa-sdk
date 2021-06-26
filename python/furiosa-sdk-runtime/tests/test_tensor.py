import unittest
import random

import numpy as np
import tensorflow as tf

from furiosa.runtime.tensor import TensorArray, numpy_dtype, DataType
from tests.test_base import MNIST_MOBINENET_V2, SessionTester, assert_tensors_equal


class TestTensor(unittest.TestCase):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    @classmethod
    def setUpClass(cls):
        cls.sess = SessionTester(MNIST_MOBINENET_V2).session

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_dtype(self):
        tensor = self.sess.input(0)
        self.assertEqual(np.uint8, numpy_dtype(tensor.dtype()))

    def test_tensor_desc(self):
        tensor = self.sess.input(0)
        self.assertEqual(4, tensor.ndim())
        self.assertEqual("NHWC", tensor.format())
        self.assertEqual((1, 28, 28, 1), tensor.shape())
        self.assertEqual(DataType.UINT8, tensor.dtype())
        self.assertEqual(np.uint8, numpy_dtype(tensor))
        self.assertEqual(784, tensor.length())
        self.assertEqual(
            tensor.__repr__(),
            "TensorDesc: shape=(1, 28, 28, 1), dtype=uint8, format=NHWC, size=784, len=784",
        )

    def test_tensor(self):
        sess = self.sess
        self.assertEqual(sess.input_num(), 1)
        self.assertEqual(sess.output_num(), 1)

        inputs: TensorArray = sess.allocate_inputs()
        inputs2: TensorArray = sess.allocate_inputs()
        self.assertEqual(1, len(inputs))
        self.assertEqual(1, len(inputs[0:1]))
        self.assertEqual(np.uint8, numpy_dtype(inputs[0]))

        # Testing TensorArray.__repr__()
        inputs[0] = np.zeros(inputs[0].shape(), np.uint8)
        self.assertTrue(
            inputs[0]
            .__repr__()
            .startswith("<Tensor: shape=(1, 28, 28, 1), dtype=DataType.UINT8, numpy=[[[[0]")
        )

        for i in range(0, 10):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.x_test[idx : idx + 1]

            inputs[0] = ndarray_value
            assert_tensors_equal(ndarray_value, inputs[0].numpy())

            # idempotent
            self.assertEqual(inputs[0], inputs[0])

            # equality of contents
            inputs2[0] = ndarray_value
            self.assertEqual(inputs[0], inputs2[0])


if __name__ == "__main__":
    unittest.main()
