"""
Common fixtures to be used from all tests.

See https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from typing import List

import pytest
from furiosa.registry import Artifact, ModelMetadata, RuntimeConfig


@pytest.fixture(scope="module")
def artifacts() -> List[Artifact]:
    return [
        Artifact(
            name="mlcommons_resnet50_v1.5_int8",
            family="ResNet",
            location="https://github.com/furiosa-ai/npu-models/blob/master/mlcommons/mlcommons_resnet50_v1.5_int8.onnx",
            format="onnx",
            description="ResNet v1.5 model for MLCommons",
            config=RuntimeConfig(
                npu_device="npu0pe0",
                compiler_config={"keep_unsignedness": True, "split_unit": 0},
            ),
            metadata=ModelMetadata(
                arxiv="https://arxiv.org/abs/1512.03385.pdf", year=None, month=None
            ),
        ),
        Artifact(
            name="mlcommons_ssd_mobilenet_v1_int8",
            family="MobileNetV1",
            location="https://github.com/furiosa-ai/npu-models/blob/master/mlcommons/mlcommons_ssd_mobilenet_v1_int8.onnx",
            format="onnx",
            description="MobileNet v1 model for MLCommons",
            config=RuntimeConfig(npu_device="npu0pe0", compiler_config=None),
            metadata=ModelMetadata(
                arxiv="https://arxiv.org/abs/1704.04861.pdf", year=None, month=None
            ),
        ),
    ]


@pytest.fixture(scope="module")
def model_file() -> str:
    return "./tests/fixtures/models/MNISTnet_uint8_quant_without_softmax.tflite"


@pytest.fixture(scope="module")
def artifact_file() -> str:
    return "./tests/fixtures/artifact.toml"


@pytest.fixture(scope="module")
def MNISTnet(model_file) -> bytes:
    with open(model_file, "rb") as data:
        return data.read()
