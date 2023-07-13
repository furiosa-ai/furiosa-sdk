from pathlib import Path

import pytest

root = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def compiler_config() -> Path:
    return root / "tests" / "data" / "compiler_config.yaml"


@pytest.fixture
def invalid_compiler_config() -> Path:
    return root / "tests" / "data" / "invalid_compiler_config.yaml"


@pytest.fixture
def quantized_mnist_tflite() -> Path:
    return root / "tests" / "data" / "MNISTnet_uint8_quant_without_softmax.tflite"
