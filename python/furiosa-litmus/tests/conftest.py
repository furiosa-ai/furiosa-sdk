from pathlib import Path

import pytest

root = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def mnist_onnx() -> Path:
    return root / "tests" / "data" / "mnist_8.onnx"


@pytest.fixture
def test_onnx() -> Path:
    return root / "tests" / "data" / "test.onnx"


@pytest.fixture
def mnist_tflite() -> Path:
    return root / "tests" / "data" / "MNISTnet_uint8_quant_without_softmax.tflite"
