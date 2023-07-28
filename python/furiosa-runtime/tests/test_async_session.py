import os
from pathlib import Path
import random

import pytest

from furiosa.runtime import errors, session


def test_run_async(mnist_onnx, mnist_images):
    submitter, receiver = session.create_async(model=Path(mnist_onnx), input_queue_size=50)

    count = 50
    for i in range(count):
        j = random.randrange(0, 9999, 1)
        images = mnist_images[j : j + 1]

        assert images.shape == submitter.input(0).shape
        assert images.dtype == submitter.input(0).dtype.numpy_dtype

        submitter.submit([images], i)

    assert {next(receiver)[0] for _ in range(count)} == set(range(count))

    assert submitter.close()
    assert receiver.close()


def test_create(mnist_onnx):
    submitter, receiver = session.create_async(
        mnist_onnx,
        worker_num=1,
        input_queue_size=1,
        output_queue_size=1,
        compiler_config={"allow_precision_error": True},
    )

    assert submitter.close()
    assert receiver.close()


def test_timeout(mnist_onnx):
    submitter, receiver = session.create_async(model=mnist_onnx)

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(Exception):
        receiver.recv(timeout=int(0))
    with pytest.raises(Exception):
        receiver.recv(timeout=100)

    assert submitter.close()

    # Recv after close
    with pytest.raises(Exception):
        receiver.recv()
    with pytest.raises(Exception):
        receiver.recv(timeout=0)
    with pytest.raises(Exception):
        receiver.recv(timeout=100)

    assert receiver.close()


def test_submitter_closed(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx)
    assert submitter.close()

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(Exception):
        submitter.inputs()
    with pytest.raises(Exception):
        submitter.input_num()
    with pytest.raises(Exception):
        submitter.outputs()
    with pytest.raises(Exception):
        submitter.output_num()
    with pytest.raises(Exception):
        submitter.summary()

    assert receiver.close()


def test_queue_closed(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx)
    assert receiver.close()
    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(Exception):
        receiver.recv()

    assert submitter.close()


@pytest.mark.skipif(os.getenv("NPU_DEVNAME") is None, reason="No NPU_DEVNAME defined")
def test_device_busy(mnist_onnx):
    submitter, receiver = session.create_async(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    # FIXME: Add specific error variants to furiosa-native-runtime
    with pytest.raises(errors.NativeException):
        session.create_async(mnist_onnx, device=os.getenv("NPU_DEVNAME"))

    assert submitter.close()
    assert receiver.close()
