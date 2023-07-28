import glob
import os

import pytest

from furiosa.litmus import validate


def test_test_model(test_onnx):
    assert validate(test_onnx, "archive", False, False, "warboy-2pe")
    zip_file = glob.glob("archive-*.zip")
    assert len(zip_file) > 0
    os.remove(zip_file[0])


def test_mnist_model(mnist_onnx):
    assert validate(mnist_onnx, None, False, False, "warboy-2pe")


def test_tflite_model(mnist_tflite):
    with pytest.raises(SystemExit):
        validate(mnist_tflite, None, False, False, "warboy-2pe")
