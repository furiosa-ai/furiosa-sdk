import random
import unittest

import numpy as np
import mnist

from furiosa.runtime import session, errors
from tests.test_base import MNIST_ONNX, SessionTester, ensure_test_device, NAMED_TENSORS_ONNX

NPU_DEVICE_READY = ensure_test_device()


class TestSession(unittest.TestCase):
    mnist_images = mnist.train_images().reshape((60000, 1, 28, 28)).astype(np.float32)

    @classmethod
    def setUpClass(cls):
        cls.tester = SessionTester(MNIST_ONNX)

    @classmethod
    def tearDownClass(cls):
        cls.tester.close()

    def test_run(self):
        sess = self.tester.session

        self.assertEqual(sess.summary(),
                         """\
Inputs:
{0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
Outputs:
{0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}""")

        items = 50
        for _ in range(0, items):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.mnist_images[idx:idx + 1]
            self.assertEqual(ndarray_value.shape, sess.input(0).shape)
            self.assertEqual(ndarray_value.dtype, sess.input(0).dtype.numpy_dtype)

            result1 = sess.run(ndarray_value)
            result2 = sess.run([ndarray_value])
            self.assertTrue(np.array_equal(result1.numpy(), result2.numpy()))

    def test_run_invalid_input(self):
        sess = self.tester.session
        self.assertRaises(errors.InvalidInput, lambda: sess.run(np.zeros((1, 28, 28,1), dtype=np.float16)));
        self.assertRaises(errors.InvalidInput, lambda: sess.run(np.zeros((2, 28, 28, 2),
                                                                         dtype=sess.input(0).dtype.numpy_dtype)));
        self.assertTrue(sess.run([np.zeros((1, 1, 28, 28), np.float32)]));
        self.assertRaises(errors.InvalidInput, lambda: sess.run([
            np.zeros((1, 28, 28, 1), np.uint8),
            np.zeros((1, 28, 28, 1), np.uint8)
        ]));


@unittest.skipIf(not NPU_DEVICE_READY, "No NPU device")
class TestDeviceBusy(unittest.TestCase):
    def test_device_busy(self):
        with session.create(MNIST_ONNX) as sess:
            self.assertRaises(errors.DeviceBusy, lambda: session.create(MNIST_ONNX))
            pass


class TestSessionOptions(unittest.TestCase):
    def test_create(self):
        with session.create(MNIST_ONNX,
                              worker_num=1,
                              compile_config = {"split_after_lower": True}) as sess:
            pass


class TestSessionWithNames(unittest.TestCase):
    def test_run_with(self):
        with session.create(NAMED_TENSORS_ONNX) as sess:
            inputs = sess.inputs()
            sess.run_with(["18_dequantized", "19_dequantized", "15_dequantized"], {
                "input.1": np.zeros(inputs[0].shape, dtype=np.single),
                "input.3": np.zeros(inputs[1].shape, dtype=np.single),
                "input": np.zeros(inputs[2].shape, dtype=np.single)
            })


if __name__ == '__main__':
    unittest.main()
