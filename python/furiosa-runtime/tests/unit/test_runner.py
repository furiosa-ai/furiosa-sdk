import random
from textwrap import dedent
from typing import List

import numpy as np
import pytest

from furiosa.runtime import is_legacy

if is_legacy:
    pytest.skip("skipping legacy tests in test_runner", allow_module_level=True)

from furiosa.runtime import FuriosaRuntimeError, Runtime, create_runner, tensor


async def run_inner(runner, mnist_images):
    model = runner.model
    summary = dedent(
        """\
        Inputs:
        {0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
        Outputs:
        {0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}
    """  # noqa: E501
    ).strip()

    assert model.summary() == summary

    count = 50
    for _ in range(count):
        idx = random.randrange(0, 9999, 1)
        images = mnist_images[idx : idx + 1]

        assert images.shape == model.input(0).shape
        assert images.dtype == model.input(0).dtype.numpy

        result1 = await runner.run(images)
        result2 = await runner.run([images])
        result3 = await runner.run_with(["Plus214_Output_0"], {"Input3": images})

        assert np.array_equal(result1, result2)
        assert np.array_equal(result1, result3)


@pytest.mark.asyncio
async def test_create_runner(mnist_onnx, mnist_images):
    async with create_runner(mnist_onnx) as runner:
        await run_inner(runner, mnist_images)


@pytest.mark.asyncio
async def test_runtime(mnist_onnx, mnist_images):
    async with Runtime() as runtime:
        async with runtime.create_runner(mnist_onnx) as runner:
            await run_inner(runner, mnist_images)


@pytest.mark.asyncio
async def test_buffer_lifetime(mnist_onnx, mnist_images):
    async with create_runner(mnist_onnx) as runner:
        # Try to unpack TensorArray to validate Tensor, TensorArray lifetime
        input1, *_ = await runner.run(mnist_images[0:1])
        input1.view()


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

        with pytest.raises(TypeError):
            await runner.run_with(None, {"Input3": np.zeros((1, 28, 28, 1), np.uint8)})


@pytest.mark.asyncio
async def test_run_with(named_tensors_onnx):
    async def assert_named_tensors(runner, input_indices: List[int], output_indices: List[int]):
        model = runner.model
        inputs = [model.input(i) for i in input_indices]
        output_names = [model.output(i).name for i in output_indices]
        outputs = await runner.run_with(
            output_names,
            {
                inputs[0].name: tensor.rand(inputs[0]),
                inputs[1].name: tensor.rand(inputs[1]),
                inputs[2].name: tensor.rand(inputs[2]),
            },
        )

        for i in range(len(outputs)):
            assert model.output(output_indices[i]).shape == outputs[i].shape
            assert model.output(output_indices[i]).dtype.numpy == outputs[i].dtype

    async with create_runner(named_tensors_onnx) as runner:
        model = runner.model
        expected = ["input.1", "input.3", "input"]
        assert model.input_num == 3
        for i in range(3):
            assert model.input(i).name == expected[i]

        assert model.output_num == 3
        expected = ["18_dequantized", "19_dequantized", "15_dequantized"]
        for i in range(3):
            assert model.output(i).name == expected[i]

        await assert_named_tensors(runner, [0, 1, 2], [0, 1, 2])
        await assert_named_tensors(runner, [2, 0, 1], [0, 1, 2])
        await assert_named_tensors(runner, [1, 2, 0], [0, 1, 2])

        await assert_named_tensors(runner, [2, 1, 0], [0, 2, 1])
        await assert_named_tensors(runner, [2, 1, 0], [1, 2, 0])
        await assert_named_tensors(runner, [2, 1, 0], [2, 0, 1])
        await assert_named_tensors(runner, [2, 1, 0], [2, 1, 0])


@pytest.mark.asyncio
async def test_run_with_invalid_names(mnist_onnx):
    async with create_runner(mnist_onnx) as runner:
        model = runner.model
        assert model.input(0).name == "Input3"
        assert model.output(0).name == "Plus214_Output_0"

        await runner.run_with(["Plus214_Output_0"], {"Input3": tensor.zeros(model.input(0))})

        with pytest.raises(ValueError):
            await runner.run_with(["WrongOutput"], {"Input3": tensor.zeros(model.input(0))})

        # Wrong input
        with pytest.raises(ValueError):
            await runner.run_with(
                ["Plus214_Output_0"], {"WrongInput3": tensor.zeros(model.input(0))}
            )


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
