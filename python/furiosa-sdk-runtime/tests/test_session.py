import random
import unittest

import numpy as np
import tensorflow as tf
from furiosa.runtime import session, errors
from tests.test_base import MNIST_MOBINENET_V2, SessionTester, ensure_test_device

NPU_DEVICE_READY = ensure_test_device()


class TestSession(unittest.TestCase):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    @classmethod
    def setUpClass(cls):
        cls.tester = SessionTester(MNIST_MOBINENET_V2)

    @classmethod
    def tearDownClass(cls):
        cls.tester.close()

    def test_run(self):
        sess = self.tester.session

        self.assertEqual("""Inputs:
{0: TensorDesc: shape=(1, 28, 28, 1), dtype=UINT8, format=NHWC, size=784, len=784}
Outputs:
{0: TensorDesc: shape=(1, 1, 1, 10), dtype=UINT8, format=NHWC, size=10, len=10}""", sess.summary())

        idx = random.randrange(0, 9999, 1)
        ndarray_value = self.x_test[idx:idx + 1]

        result1 = sess.run(ndarray_value)
        result2 = sess.run([ndarray_value])
        np.array_equal(result1.numpy(), result2.numpy())

    def test_run_invalid_input(self):
        sess = self.tester.session
        self.assertRaises(errors.InvalidInput, lambda: sess.run(np.zeros((1, 28, 28,1), dtype=np.float16)));
        self.assertRaises(errors.InvalidInput, lambda: sess.run(np.zeros((2, 28, 28, 2),
                                                                         dtype=sess.input(0).dtype.numpy_dtype)));
        self.assertTrue(sess.run([np.zeros((1, 28, 28, 1), np.uint8)]));
        self.assertRaises(errors.InvalidInput, lambda: sess.run([
            np.zeros((1, 28, 28, 1), np.uint8),
            np.zeros((1, 28, 28, 1), np.uint8)
        ]));


@unittest.skipIf(not NPU_DEVICE_READY, "No NPU device")
class TestDeviceBusy(unittest.TestCase):
    def test_device_busy(self):
        with session.create(MNIST_MOBINENET_V2) as sess:
            self.assertRaises(errors.DeviceBusy, lambda: session.create(MNIST_MOBINENET_V2))
            pass


class TestSessionOptions:
    def test_create(self):
        with session.create(MNIST_MOBINENET_V2,
                              worker_num=1,
                              compile_config = {"split_after_lower": True}) as sess:
            pass


if __name__ == '__main__':
    unittest.main()
