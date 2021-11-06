import random
import unittest

import numpy as np
import mnist

from furiosa.runtime import session, errors
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

        items = 50
        for i in range(0, items):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.mnist_images[idx:idx + 1]
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
            sess, queue = session.create_async(MNIST_ONNX,
                                               worker_num=1,
                                               input_queue_size=1,
                                               output_queue_size=1,
                                               compile_config={"split_after_lower": True})
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
            nux_sess.close()
            self.assertRaises(errors.SessionTerminated, lambda: nux_queue.recv())
            self.assertRaises(errors.SessionTerminated, lambda: nux_queue.recv(timeout=100))
        except:
            if nux_sess:
                nux_sess.close()
            if nux_queue:
                nux_queue.close();


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
