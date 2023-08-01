# Tests for legacy API `AsyncSession`
import os
from pathlib import Path
import random
from textwrap import dedent

import pytest

from furiosa.runtime import errors, session


def test_run_async(mnist_onnx, mnist_images):
    submitter, receiver = session.create_async(model=Path(mnist_onnx), input_queue_size=50)

    summary = dedent(
        """\
        Inputs:
        {0: TensorDesc(name="Input3", shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
        Outputs:
        {0: TensorDesc(name="Plus214_Output_0", shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}
    """  # noqa: E501
    ).strip()
    assert submitter.summary() == summary

    count = 50
    for i in range(count):
        j = random.randrange(0, 9999, 1)
        images = mnist_images[j : j + 1]

        assert images.shape == submitter.input(0).shape
        assert images.dtype == submitter.input(0).dtype.numpy_dtype

        submitter.submit([images], i)

    assert {next(receiver)[0] for _ in range(count)} == set(range(count))

    submitter.close()
    receiver.close()


def test_create(mnist_onnx):
    submitter, receiver = session.create_async(
        mnist_onnx,
        worker_num=1,
        input_queue_size=1,
        output_queue_size=1,
        compiler_config={"allow_precision_error": True},
    )

    submitter.close()
    receiver.close()


def test_timeout(mnist_onnx):
    submitter, receiver = session.create_async(model=mnist_onnx)

    with pytest.raises(errors.QueueWaitTimeout):
        receiver.recv(timeout=int(0))
    with pytest.raises(errors.QueueWaitTimeout):
        receiver.recv(timeout=100)

    submitter.close()

    with pytest.raises(errors.SessionTerminated):
        receiver.recv()
    with pytest.raises(errors.SessionTerminated):
        receiver.recv(timeout=0)
    with pytest.raises(errors.SessionTerminated):
        receiver.recv(timeout=100)

    receiver.close()


def test_submitter_closed(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx)
    submitter.close()

    with pytest.raises(errors.SessionClosed):
        submitter.inputs()
    with pytest.raises(errors.SessionClosed):
        submitter.input_num()
    with pytest.raises(errors.SessionClosed):
        submitter.outputs()
    with pytest.raises(errors.SessionClosed):
        submitter.output_num()
    with pytest.raises(errors.SessionClosed):
        submitter.summary()

    receiver.close()


def test_queue_closed(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx)
    receiver.close()
    with pytest.raises(errors.SessionClosed):
        receiver.recv()

    submitter.close()


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
def test_device_busy(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    with pytest.raises(errors.DeviceBusy):
        submitter, receiver = session.create_async(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    submitter.close()
    receiver.close()
