from pathlib import Path

import mnist
import numpy as np
import pytest

root = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def mnist_onnx() -> Path:
    return root / "tests" / "data" / "mnist_8.onnx"


@pytest.fixture
def named_tensors_onnx() -> Path:
    return root / "tests" / "data" / "named_tensors.onnx"


@pytest.fixture
def quantized_conv_truncated_onnx() -> Path:
    return root / "tests" / "data" / "quantized_i8_conv_truncated.onnx"


@pytest.fixture
def mnist_images():
    return mnist.train_images().reshape((60000, 1, 28, 28)).astype(np.float32)
