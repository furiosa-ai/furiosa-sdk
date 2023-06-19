"""
Common fixtures to be used from all tests.
"""

import pytest


@pytest.fixture(scope="module")
def model_file() -> str:
    return "./tests/unit/fixtures/MNISTnet_uint8_quant_without_softmax.tflite"


@pytest.fixture(scope="module")
def model_binary(model_file) -> bytes:
    with open(model_file, "rb") as data:
        return data.read()
