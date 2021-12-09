import random
from typing import List
import unittest

import mnist
import numpy as np

from furiosa.runtime import errors, session
from tests.test_base import MNIST_ONNX, NAMED_TENSORS_ONNX, SessionTester, ensure_test_device

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

        self.assertEqual(
            sess.summary(),
            """\
Inputs:
{0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
Outputs:
{0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}""",
        )

        items = 50
        for _ in range(0, items):
            idx = random.randrange(0, 9999, 1)
            ndarray_value = self.mnist_images[idx : idx + 1]
            self.assertEqual(ndarray_value.shape, sess.input(0).shape)
            self.assertEqual(ndarray_value.dtype, sess.input(0).dtype.numpy_dtype)

            result1 = sess.run(ndarray_value)
            result2 = sess.run([ndarray_value])
            self.assertTrue(np.array_equal(result1.numpy(), result2.numpy()))

            result3 = sess.run_with(["Plus214_Output_0"], {"Input3": ndarray_value})
            self.assertTrue(np.array_equal(result1.numpy(), result3.numpy()))

    def test_run_invalid_input(self):
        sess = self.tester.session
        self.assertRaises(
            errors.InvalidInput, lambda: sess.run(np.zeros((1, 28, 28, 1), dtype=np.float16))
        )
        self.assertRaises(
            errors.InvalidInput,
            lambda: sess.run(np.zeros((2, 28, 28, 2), dtype=sess.input(0).dtype.numpy_dtype)),
        )
        self.assertTrue(sess.run([np.zeros((1, 1, 28, 28), np.float32)]))
        self.assertRaises(
            errors.InvalidInput,
            lambda: sess.run(
                [np.zeros((1, 28, 28, 1), np.uint8), np.zeros((1, 28, 28, 1), np.uint8)]
            ),
        )
        self.assertRaises(
            errors.InvalidInput,
            lambda: sess.run_with(None, {"Input3": np.zeros((1, 28, 28, 1), np.uint8)}),
        )


@unittest.skipIf(not NPU_DEVICE_READY, "No NPU device")
class TestDeviceBusy(unittest.TestCase):
    def test_device_busy(self):
        with session.create(MNIST_ONNX) as sess:
            self.assertRaises(errors.DeviceBusy, lambda: session.create(MNIST_ONNX))
            pass


class TestSessionOptions(unittest.TestCase):
    def test_create(self):
        with session.create(
            MNIST_ONNX, worker_num=1, compile_config={"split_after_lower": True}
        ) as sess:
            pass


class TestSessionWithNames(unittest.TestCase):
    def assert_named_tensors(self, sess, input_indices: List[int], output_indices: List[int]):
        from furiosa.runtime import tensor

        inputs = [sess.input(idx) for idx in input_indices]
        output_names = [sess.output(idx).name for idx in output_indices]
        outputs = sess.run_with(
            output_names,
            {
                inputs[0].name: tensor.rand(inputs[0]),
                inputs[1].name: tensor.rand(inputs[1]),
                inputs[2].name: tensor.rand(inputs[2]),
            },
        )

        for idx in range(0, len(outputs)):
            self.assertEqual(sess.output(output_indices[idx]).shape, outputs[idx].shape)
            self.assertEqual(sess.output(output_indices[idx]).dtype, outputs[idx].dtype)

    def test_run_with(self):
        with session.create(NAMED_TENSORS_ONNX) as sess:
            expected_inputs = ["input.1", "input.3", "input"]
            self.assertEqual(sess.input_num, 3)
            for idx in range(0, sess.input_num):
                self.assertEqual(sess.input(idx).name, expected_inputs[idx])

            self.assertEqual(sess.output_num, 3)
            expected_outputs = ["18_dequantized", "19_dequantized", "15_dequantized"]
            for idx in range(0, sess.output_num):
                self.assertEqual(sess.output(idx).name, expected_outputs[idx])

            self.assert_named_tensors(sess, [0, 1, 2], [0, 1, 2])
            self.assert_named_tensors(sess, [2, 0, 1], [0, 1, 2])
            self.assert_named_tensors(sess, [1, 2, 0], [0, 1, 2])

            self.assert_named_tensors(sess, [2, 1, 0], [0, 2, 1])
            self.assert_named_tensors(sess, [2, 1, 0], [1, 2, 0])
            self.assert_named_tensors(sess, [2, 1, 0], [2, 0, 1])
            self.assert_named_tensors(sess, [2, 1, 0], [2, 1, 0])

    def test_run_with_invalid_names(self):
        from furiosa.runtime import tensor

        with session.create(MNIST_ONNX) as sess:
            self.assertEqual(sess.input(0).name, "Input3")
            self.assertEqual(sess.output(0).name, "Plus214_Output_0")
            sess.run_with(["Plus214_Output_0"], {"Input3": tensor.zeros(sess.input(0))})

            # Wrong output
            self.assertRaises(
                errors.InvalidInput,
                lambda: sess.run_with(["WrongOutput"], {"Input3": tensor.zeros(sess.input(0))}),
            )
            # Wrong input
            self.assertRaises(
                errors.InvalidInput,
                lambda: sess.run_with(
                    ["Plus214_Output_0"], {"WrongInput3": tensor.zeros(sess.input(0))}
                ),
            )


if __name__ == '__main__':
    unittest.main()
