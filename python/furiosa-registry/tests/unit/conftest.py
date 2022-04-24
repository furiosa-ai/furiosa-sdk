"""
Common fixtures to be used from all tests.

See https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import pytest


@pytest.fixture(scope="module")
def model_file() -> str:
    return "./tests/unit/fixtures/models/MNISTnet_uint8_quant_without_softmax.tflite"


@pytest.fixture(scope="module")
def model_binary(model_file) -> bytes:
    with open(model_file, "rb") as data:
        return data.read()
