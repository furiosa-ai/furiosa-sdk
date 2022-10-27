import random
import unittest

import mnist
import numpy as np

from furiosa.runtime import session
from furiosa.runtime.tensor import DataType, TensorArray, numpy_dtype
from tests.test_base import (
    MNIST_ONNX,
    NAMED_TENSORS_ONNX,
    QUANTIZED_CONV_TRUNCATED_TEST,
    SessionTester,
    assert_tensors_equal,
)


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


class TestLoweredTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler_config = {
            "remove_unlower": True,
        }
        cls.sess = SessionTester(
            QUANTIZED_CONV_TRUNCATED_TEST, compiler_config=compiler_config
        ).session

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_tensor_desc(self):
        tensor = self.sess.output(0)
        self.assertEqual(6, tensor.ndim)
        self.assertEqual(DataType.INT8, tensor.dtype)
        self.assertEqual(np.int8, numpy_dtype(tensor))
        self.assertLess(tensor.length * np.dtype(tensor.numpy_dtype).itemsize, tensor.size)

    def test_tensor_numpy(self):
        tensor = self.sess.output(0)

        strides = []
        stride = 1
        for axis in reversed(range(tensor.ndim)):
            strides.append(stride)
            stride *= tensor.dim(axis)
        strides = tuple(reversed(strides))

        inputs = []
        for session_input in self.sess.inputs():
            inputs.append(np.random.random(session_input.shape).astype(np.float32))
        outputs = self.sess.run(inputs)

        output = outputs[0].view()
        self.assertEqual(6, output.ndim)
        self.assertEqual(np.int8, numpy_dtype(output))
        self.assertNotEqual(strides, output.strides)

        # calling numpy() rearranges the memory layout due to internal copy() call
        output = outputs[0].numpy()
        self.assertEqual(6, output.ndim)
        self.assertEqual(np.int8, numpy_dtype(output))
        self.assertEqual(strides, output.strides)


if __name__ == '__main__':
    unittest.main()
