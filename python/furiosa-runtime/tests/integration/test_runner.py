import random

import numpy as np
import pytest

from furiosa.runtime import FuriosaRuntimeError, Runtime, create_runner

RUN_COUNT = 50


@pytest.mark.asyncio
async def test_create_runner(mnist_onnx, mnist_images):
    async with create_runner(mnist_onnx) as runner:
        # FIXME: Add named tensor test
        for _ in range(RUN_COUNT):
            idx = random.randrange(0, 9999, 1)
            images = mnist_images[idx : idx + 1]

            assert images.shape == runner.model.input(0).shape
            assert images.dtype == runner.model.input(0).dtype.numpy

            result1 = await runner.run(images)
            result2 = await runner.run([images])

            assert np.array_equal(result1, result2)


@pytest.mark.asyncio
async def test_runtime(mnist_onnx, mnist_images):
    async with Runtime() as runtime:
        async with runtime.create_runner(mnist_onnx) as runner:
            # FIXME: Add named tensor test
            for _ in range(RUN_COUNT):
                idx = random.randrange(0, 9999, 1)
                images = mnist_images[idx : idx + 1]

                assert images.shape == runner.model.input(0).shape
                assert images.dtype == runner.model.input(0).dtype.numpy

                result1 = await runner.run(images)
                result2 = await runner.run([images])

                assert np.array_equal(result1, result2)


@pytest.mark.asyncio
async def test_run_invalid_input(mnist_onnx):
    async with create_runner(mnist_onnx) as runner:
        # FIXME: Add specific error variants to furiosa-native-runtime
        with pytest.raises(TypeError):
            await runner.run(np.zeros((1, 28, 28, 1), dtype=np.float16))

        with pytest.raises(ValueError):
            await runner.run(np.zeros((2, 28, 28, 2), dtype=runner.model.input(0).dtype.numpy))

        assert await runner.run([np.zeros((1, 1, 28, 28), np.float32)])

        with pytest.raises(ValueError):
            await runner.run(
                [np.zeros((1, 28, 28, 1), np.uint8), np.zeros((1, 28, 28, 1), np.uint8)]
            )

    # FIXME: Add Named tensor test


@pytest.mark.asyncio
async def test_device_busy(mnist_onnx):
    runner = await create_runner(mnist_onnx)

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        await create_runner(mnist_onnx)

    assert await runner.close()


@pytest.mark.asyncio
async def test_closed(mnist_onnx):
    runner = await create_runner(mnist_onnx)
    assert await runner.close()

    with pytest.raises(FuriosaRuntimeError):
        await runner.run([np.zeros((1, 1, 28, 28), np.float32)])


@pytest.mark.asyncio
async def test_create(mnist_onnx):
    runner = await create_runner(
        mnist_onnx, worker_num=1, compiler_config={"allow_precision_error": True}
    )
    assert runner
    assert await runner.close()


@pytest.mark.asyncio
async def test_batch(mnist_onnx):
    async with create_runner(mnist_onnx, batch_size=1) as runner:
        assert runner.model.input(0).shape == (1, 1, 28, 28)

    async with create_runner(mnist_onnx, batch_size=4) as runner:
        assert runner.model.input(0).shape == (4, 1, 28, 28)


@pytest.mark.asyncio
async def test_invalid_config_name(mnist_onnx):
    invalid_config = {"INVALID_COMPILER_CONFIG_NAME": True}
    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        await create_runner(mnist_onnx, worker_num=1, compiler_config=invalid_config)


@pytest.mark.asyncio
async def test_invalid_config_type(mnist_onnx):
    invalid_config = {"remove_lower": 1}  # Correct value type is boolean
    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(FuriosaRuntimeError):
        await create_runner(mnist_onnx, worker_num=1, compiler_config=invalid_config)
