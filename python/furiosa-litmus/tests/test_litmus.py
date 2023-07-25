import os

from furiosa.litmus import validate
from furiosa.litmus.reporter import Reporter


def test_test_model(test_onnx):
    reporter = Reporter("archive")
    err = validate(test_onnx, reporter, False, False, "warboy-2pe")
    assert not isinstance(err, BaseException)
    assert os.path.exists(reporter.tmp_dir.name)


def test_mnist_model(mnist_onnx):
    err = validate(mnist_onnx, Reporter(None), False, False, "warboy-2pe")
    assert not isinstance(err, BaseException)


def test_tflite_model(mnist_tflite):
    err = validate(mnist_tflite, Reporter(None), False, False, "warboy-2pe")
    assert isinstance(err, SystemExit)
