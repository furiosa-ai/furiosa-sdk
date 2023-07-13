import pytest

from furiosa.litmus import validate


def test_test_model(test_onnx):
    assert validate(test_onnx, False, "warboy-2pe")


def test_mnist_model(mnist_onnx):
    assert validate(mnist_onnx, False, "warboy-2pe")


def test_tflite_model(mnist_tflite):
    with pytest.raises(SystemExit):
        validate(mnist_tflite, False, "warboy-2pe")
