import os
import random

import numpy as np
import pytest

from furiosa.runtime import is_legacy, session


def test_run(mnist_onnx, mnist_images):
    with session.create(model=mnist_onnx) as runner:
        # FIXME: Add named tesnor test
        count = 50
        for _ in range(count):
            idx = random.randrange(0, 9999, 1)
            images = mnist_images[idx : idx + 1]

            assert images.shape == runner.input(0).shape
            assert images.dtype == runner.input(0).dtype.numpy_dtype

            result1 = runner.run(images)
            result2 = runner.run([images])

            assert np.array_equal(result1.numpy(), result2.numpy())


@pytest.mark.skipif(is_legacy, reason="Legacy runtime has segfault buffer is invalid after unpack")
def test_buffer_lifetime(mnist_onnx, mnist_images):
    with session.create(model=mnist_onnx) as runner:
        # Try to unpack TensorArray to validate Tensor, TensorArray lifetime
        input1, *_ = runner.run(mnist_images[0:1])
        input1.view()


def test_run_invalid_input(mnist_onnx):
    with session.create(model=mnist_onnx) as runner:
        # FIXME: Add specific error variants to furiosa-native-runtime
        with pytest.raises(Exception):
            runner.run(np.zeros((1, 28, 28, 1), dtype=np.float16))

        with pytest.raises(Exception):
            runner.run(np.zeros((2, 28, 28, 2), dtype=runner.input(0).dtype.numpy_dtype))

        assert runner.run([np.zeros((1, 1, 28, 28), np.float32)])

        with pytest.raises(Exception):
            runner.run([np.zeros((1, 28, 28, 1), np.uint8), np.zeros((1, 28, 28, 1), np.uint8)])

    # FIXME: Add Named tensor test


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
def test_device_busy(mnist_onnx):
    runner = session.create(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(Exception):
        session.create(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    runner.close()


def test_closed(mnist_onnx):
    runner = session.create(mnist_onnx)
    runner.close()

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(Exception):
        runner.inputs()
    with pytest.raises(Exception):
        runner.input_num()
    with pytest.raises(Exception):
        runner.outputs()
    with pytest.raises(Exception):
        runner.output_num()
    with pytest.raises(Exception):
        runner.summary()
    with pytest.raises(Exception):
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
    # FIXME: Add specific error variants to furiosa-native-runtime

    with pytest.raises(Exception):
        session.create(mnist_onnx, compiler_config=invalid_config)
