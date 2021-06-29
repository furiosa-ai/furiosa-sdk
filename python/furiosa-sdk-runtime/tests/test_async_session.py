import random
import unittest

import numpy as np
import tensorflow as tf

from tests.test_base import MNIST_MOBINENET_V2, AsyncSessionTester


class TestAsyncSession(unittest.TestCase):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    @classmethod
    def setUpClass(cls):
        cls.tester = AsyncSessionTester(MNIST_MOBINENET_V2)

    @classmethod
    def tearDownClass(cls):
        cls.tester.close()

    def test_run_async(self):
        async_sess = self.tester.session

        items = 50
        for i in range(0, items):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.x_test[idx : idx + 1]
            async_sess.submit([ndarray_value], i)

        keys = set()
        for key, _ in self.tester.queue:
            keys.add(key)

            if len(keys) == items:
                break

        self.assertEqual(set(range(0, items)), keys)


if __name__ == "__main__":
    unittest.main()
