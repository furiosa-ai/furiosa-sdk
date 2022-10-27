import random
import unittest

import mnist
import numpy as np

from furiosa.runtime import errors, session
from tests.test_base import MNIST_ONNX, AsyncSessionTester, ensure_test_device

NPU_DEVICE_READY = ensure_test_device()


class TestAsyncSession(unittest.TestCase):
    mnist_images = mnist.train_images().reshape((60000, 1, 28, 28)).astype(np.float32)

    @classmethod
    def setUpClass(cls):
        cls.tester = AsyncSessionTester(MNIST_ONNX)

    @classmethod
    def tearDownClass(cls):
        cls.tester.close()

    def test_run_async(self):
        async_sess = self.tester.session

        self.assertEqual(
            async_sess.summary(),
            """\
Inputs:
{0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
Outputs:
{0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}""",
        )

        items = 50
        for i in range(0, items):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.mnist_images[idx : idx + 1]

            self.assertEqual(ndarray_value.shape, async_sess.input(0).shape)
            self.assertEqual(ndarray_value.dtype, async_sess.input(0).dtype.numpy_dtype)

            async_sess.submit([ndarray_value], i)

        keys = set()
        for key, _ in self.tester.queue:
            keys.add(key)

            if len(keys) == items:
                break

        self.assertEqual(set(range(0, items)), keys)


class TestAsyncSessionExceptions(unittest.TestCase):
    def test_create(self):
        sess = None
        queue = None
        try:
            sess, queue = session.create_async(
                MNIST_ONNX,
                worker_num=1,
                input_queue_size=1,
                output_queue_size=1,
                compiler_config={"allow_precision_error": True},
            )
        finally:
            if queue:
                queue.close()
            if sess:
                sess.close()

    def test_next_with_timeout(self):
        nux_sess = None
        nux_queue = None
        try:
            nux_sess, nux_queue = session.create_async(model=MNIST_ONNX)
            self.assertRaises(errors.QueueWaitTimeout, lambda: nux_queue.recv(timeout=100))
            self.assertRaises(errors.QueueWaitTimeout, lambda: nux_queue.recv(timeout=int(0)))
            nux_sess.close()
            self.assertRaises(errors.SessionTerminated, lambda: nux_queue.recv())
            self.assertRaises(errors.SessionTerminated, lambda: nux_queue.recv(timeout=0))
            self.assertRaises(errors.SessionTerminated, lambda: nux_queue.recv(timeout=100))
        except:
            if nux_sess:
                nux_sess.close()
            if nux_queue:
                nux_queue.close()


@unittest.skipIf(not NPU_DEVICE_READY, "No NPU device")
class TestDeviceBusy(unittest.TestCase):
    def test_device_busy(self):
        sess = None
        queue = None
        try:
            sess, queue = session.create_async(MNIST_ONNX)
            self.assertRaises(errors.DeviceBusy, lambda: session.create_async(MNIST_ONNX))
        finally:
            if sess:
                sess.close()
            if queue:
                queue.close()


if __name__ == '__main__':
    unittest.main()
