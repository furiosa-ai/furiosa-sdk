import random

import numpy as np

from furiosa.runtime import session
from furiosa.runtime.tensor import DataType, numpy_dtype


def test_dtype(mnist_onnx):
    with session.create(mnist_onnx) as runner:
        assert np.float32 == numpy_dtype(runner.input(0).dtype)


def test_tensor_desc(mnist_onnx):
    with session.create(mnist_onnx) as runner:
        tensor = runner.input(0)
        assert tensor.ndim == 4
        assert tensor.format == "NCHW"
        assert tensor.shape == (1, 1, 28, 28)
        assert tensor.dtype == DataType.FLOAT32
        assert numpy_dtype(tensor) == np.float32
        assert tensor.length == 784
        assert tensor.size == 3136

    # FIXME: Add named tensor check


def assert_tensors_equal(expected, result):
    assert np.allclose(expected, result, atol=1.0), "{} was expected, but the result was {}".format(
        expected, result
    )


def test_tensor(mnist_onnx, mnist_images):
    with session.create(mnist_onnx) as runner:
        assert runner.input_num == 1
        assert runner.output_num == 1

        inputs = runner.allocate_inputs()
        inputs2 = runner.allocate_inputs()

        assert len(inputs) == 1
        assert len(inputs[0:1]) == 1
        assert numpy_dtype(inputs[0]) == np.float32

        # FIXME: Add named tensor check

        for i in range(10):
            idx = random.randrange(0, 9999, 1)
            images = mnist_images[idx : idx + 1]

            inputs[0] = images

            assert np.allclose(images, inputs[0].numpy())

            # Idempotent
            assert inputs[0] == inputs[0]

            # Equality of contents
            inputs2[0] = images
            assert inputs[0] == inputs2[0]


def test_tensor_names(named_tensors_onnx):
    with session.create(named_tensors_onnx) as runner:
        assert runner.input_num == 3
        assert runner.output_num == 3

        # FIXME: Add named tensor check


def test_lowered_tensor_desc(quantized_conv_truncated_onnx):
    with session.create(
        quantized_conv_truncated_onnx,
        compiler_config={
            "remove_unlower": True,
        },
    ) as runner:
        tensor = runner.output(0)

        assert tensor.ndim == 6
        assert tensor.dtype == DataType.INT8
        assert numpy_dtype(tensor) == np.int8
        assert tensor.size > tensor.length * np.dtype(tensor.numpy_dtype).itemsize


def test_lowered_tensor_numpy(quantized_conv_truncated_onnx):
    with session.create(
        quantized_conv_truncated_onnx,
        compiler_config={
            "remove_unlower": True,
        },
    ) as runner:
        tensor = runner.output(0)

        strides = []
        stride = 1
        for axis in reversed(range(tensor.ndim)):
            strides.append(stride)
            stride *= tensor.dim(axis)
        strides = tuple(reversed(strides))

        inputs = []
        for input in runner.inputs():
            inputs.append(np.random.random(input.shape).astype(np.float32))

        outputs = runner.run(inputs)

        output = outputs[0].view()

        assert output.ndim == 6
        assert numpy_dtype(output) == np.int8
        assert output.strides != strides

        # Calling numpy() rearranges the memory layout due to internal copy() call
        output = outputs[0].numpy()

        assert output.ndim == 6
        assert numpy_dtype(output) == np.int8
        assert output.strides == strides
