import unittest
import random

import numpy as np
import tensorflow as tf

from tests.test_base import MNIST_MOBINENET_V2, BlockingPredictionTester, AsyncPredictionTester


class MnistMobilenetV2(unittest.TestCase):
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    def test_session(self):
        tester = BlockingPredictionTester(MNIST_MOBINENET_V2)

        for i in range(0, 10):
            idx = random.randrange(0, 9999, 1)
            tester.assert_equals(np.array(self.x_test[idx:idx + 1], dtype=np.uint8))

        tester.close()

    def test_async_session(self):
        tester = AsyncPredictionTester(MNIST_MOBINENET_V2)

        for i in range(0, 10):
            idx = random.randrange(0, 9999, 1)
            tester.assert_equals(np.array(self.x_test[idx:idx + 1], dtype=np.uint8))

        tester.close()


if __name__ == '__main__':
    unittest.main()
