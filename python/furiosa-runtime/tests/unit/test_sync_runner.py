import random

import numpy as np
import pytest

from furiosa.runtime.sync import FuriosaRuntimeError, Runtime, create_runner

RUN_COUNT = 50


def test_create_runner(mnist_onnx, mnist_images):
    with create_runner(mnist_onnx) as runner:
        # FIXME: Add named tensor test
        for _ in range(RUN_COUNT):
            idx = random.randrange(0, 9999, 1)
            images = mnist_images[idx : idx + 1]

            assert images.shape == runner.model.input(0).shape
            assert images.dtype == runner.model.input(0).dtype.numpy

            result1 = runner.run(images)
            result2 = runner.run([images])

            assert np.array_equal(result1, result2)


def test_runtime(mnist_onnx, mnist_images):
    with Runtime() as runtime:
        with runtime.create_runner(mnist_onnx) as runner:
            # FIXME: Add named tensor test
            for _ in range(RUN_COUNT):
                idx = random.randrange(0, 9999, 1)
                images = mnist_images[idx : idx + 1]

                assert images.shape == runner.model.input(0).shape
                assert images.dtype == runner.model.input(0).dtype.numpy

                result1 = runner.run(images)
                result2 = runner.run([images])

                assert np.array_equal(result1, result2)


def test_run_invalid_input(mnist_onnx):
    with create_runner(mnist_onnx) as runner:
        # FIXME: Add specific error variants to furiosa-native-runtime
        with pytest.raises(TypeError):
            runner.run(np.zeros((1, 28, 28, 1), dtype=np.float16))

        with pytest.raises(ValueError):
            runner.run(np.zeros((2, 28, 28, 2), dtype=runner.model.input(0).dtype.numpy))

        assert runner.run([np.zeros((1, 1, 28, 28), np.float32)])

        with pytest.raises(ValueError):
            runner.run([np.zeros((1, 28, 28, 1), np.uint8), np.zeros((1, 28, 28, 1), np.uint8)])

    # FIXME: Add Named tensor test


def test_device_busy(mnist_onnx):
    runner = create_runner(mnist_onnx)

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        create_runner(mnist_onnx)

    assert runner.close()


def test_closed(mnist_onnx):
    runner = create_runner(mnist_onnx)
    assert runner.close()

    with pytest.raises(FuriosaRuntimeError):
        runner.run([np.zeros((1, 1, 28, 28), np.float32)])


def test_create(mnist_onnx):
    runner = create_runner(
        mnist_onnx, worker_num=1, compiler_config={"allow_precision_error": True}
    )
    assert runner
    assert runner.close()


def test_batch(mnist_onnx):
    with create_runner(mnist_onnx, batch_size=1) as runner:
        assert runner.model.input(0).shape == (1, 1, 28, 28)

    with create_runner(mnist_onnx, batch_size=4) as runner:
        assert runner.model.input(0).shape == (4, 1, 28, 28)


def test_invalid_config_name(mnist_onnx):
    invalid_config = {"INVALID_COMPILER_CONFIG_NAME": True}
    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        create_runner(mnist_onnx, worker_num=1, compiler_config=invalid_config)


def test_invalid_config_type(mnist_onnx):
    invalid_config = {"remove_lower": 1}  # Correct value type is boolean
    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        create_runner(mnist_onnx, worker_num=1, compiler_config=invalid_config)
