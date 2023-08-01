# Tests for legacy API `Session`
import os
import random
from textwrap import dedent
from typing import List

import numpy as np
import pytest

from furiosa.runtime import errors, is_legacy, session, tensor


def test_run(mnist_onnx, mnist_images):
    with session.create(model=mnist_onnx) as runner:
        summary = dedent(
            """\
            Inputs:
            {0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
            Outputs:
            {0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}
        """  # noqa: E501
        ).strip()

        assert runner.summary() == summary

        count = 50
        for _ in range(count):
            idx = random.randrange(0, 9999, 1)
            images = mnist_images[idx : idx + 1]

            assert images.shape == runner.input(0).shape
            assert images.dtype == runner.input(0).dtype.numpy_dtype

            result1 = runner.run(images)
            result2 = runner.run([images])
            result3 = runner.run_with(["Plus214_Output_0"], {"Input3": images})

            assert np.array_equal(result1.numpy(), result3.numpy())
            assert np.array_equal(result1.numpy(), result2.numpy())


@pytest.mark.skipif(is_legacy, reason="Legacy runtime has segfault buffer is invalid after unpack")
def test_buffer_lifetime(mnist_onnx, mnist_images):
    with session.create(model=mnist_onnx) as runner:
        # Try to unpack TensorArray to validate Tensor, TensorArray lifetime
        input1, *_ = runner.run(mnist_images[0:1])
        input1.view()


def test_run_invalid_input(mnist_onnx):
    with session.create(model=mnist_onnx) as runner:
        with pytest.raises(errors.InvalidInput):
            runner.run(np.zeros((1, 28, 28, 1), dtype=np.float16))

        with pytest.raises(errors.InvalidInput):
            runner.run(np.zeros((2, 28, 28, 2), dtype=runner.input(0).dtype.numpy_dtype))

        assert runner.run([np.zeros((1, 1, 28, 28), np.float32)])

        with pytest.raises(errors.InvalidInput):
            runner.run([np.zeros((1, 28, 28, 1), np.uint8), np.zeros((1, 28, 28, 1), np.uint8)])

        # NOTE: Nux returns InvalidInput while furiosa-native-runtime returns TypeError here
        with pytest.raises((TypeError, errors.InvalidInput)):
            runner.run_with(None, {"Input3": np.zeros((1, 28, 28, 1), np.uint8)})


def test_run_with(named_tensors_onnx):
    def assert_named_tensors(runner, input_indices: List[int], output_indices: List[int]):
        inputs = [runner.input(i) for i in input_indices]
        output_names = [runner.output(i).name for i in output_indices]
        outputs = runner.run_with(
            output_names,
            {
                inputs[0].name: tensor.rand(inputs[0]),
                inputs[1].name: tensor.rand(inputs[1]),
                inputs[2].name: tensor.rand(inputs[2]),
            },
        )

        for i in range(len(outputs)):
            assert runner.output(output_indices[i]).shape == outputs[i].shape
            assert runner.output(output_indices[i]).dtype == outputs[i].dtype

    with session.create(named_tensors_onnx) as runner:
        expected = ["input.1", "input.3", "input"]
        assert runner.input_num == 3
        for i in range(3):
            assert runner.input(i).name == expected[i]

        assert runner.output_num == 3
        expected = ["18_dequantized", "19_dequantized", "15_dequantized"]
        for i in range(3):
            assert runner.output(i).name == expected[i]

        assert_named_tensors(runner, [0, 1, 2], [0, 1, 2])
        assert_named_tensors(runner, [2, 0, 1], [0, 1, 2])
        assert_named_tensors(runner, [1, 2, 0], [0, 1, 2])

        assert_named_tensors(runner, [2, 1, 0], [0, 2, 1])
        assert_named_tensors(runner, [2, 1, 0], [1, 2, 0])
        assert_named_tensors(runner, [2, 1, 0], [2, 0, 1])
        assert_named_tensors(runner, [2, 1, 0], [2, 1, 0])


def test_run_with_invalid_names(mnist_onnx):
    with session.create(mnist_onnx) as runner:
        assert runner.input(0).name == "Input3"
        assert runner.output(0).name == "Plus214_Output_0"

        runner.run_with(["Plus214_Output_0"], {"Input3": tensor.zeros(runner.input(0))})

        with pytest.raises(errors.TensorNameNotFound):
            runner.run_with(["WrongOutput"], {"Input3": tensor.zeros(runner.input(0))})

        # Wrong input
        with pytest.raises(errors.TensorNameNotFound):
            runner.run_with(["Plus214_Output_0"], {"WrongInput3": tensor.zeros(runner.input(0))})


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
def test_device_busy(mnist_onnx):
    runner = session.create(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    with pytest.raises(errors.DeviceBusy):
        session.create(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    runner.close()


def test_closed(mnist_onnx):
    runner = session.create(mnist_onnx)
    runner.close()

    with pytest.raises(errors.SessionClosed):
        runner.inputs()
    with pytest.raises(errors.SessionClosed):
        runner.input_num()
    with pytest.raises(errors.SessionClosed):
        runner.outputs()
    with pytest.raises(errors.SessionClosed):
        runner.output_num()
    with pytest.raises(errors.SessionClosed):
        runner.summary()
    with pytest.raises(errors.SessionClosed):
        runner.run(None)


def test_create(mnist_onnx):
    runner = session.create(
        mnist_onnx, worker_num=1, compiler_config={"allow_precision_error": True}
    )
    assert runner
    runner.close()


def test_batch(mnist_onnx):
    with session.create(mnist_onnx, batch_size=1) as runner:
        assert runner.input(0).shape == (1, 1, 28, 28)

    with session.create(mnist_onnx, batch_size=4) as runner:
        assert runner.input(0).shape == (4, 1, 28, 28)


def test_invalid_config(mnist_onnx):
    invalid_config = {"remove_lower": 1}  # Correct value type is boolean

    # FIXME: Add specific compiler errors to furiosa-native-runtime
    with pytest.raises(Exception):
        session.create(mnist_onnx, compiler_config=invalid_config)
