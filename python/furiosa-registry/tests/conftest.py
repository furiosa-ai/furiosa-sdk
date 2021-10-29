"""
Common fixtures to be used from all tests.

See https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from typing import List

import pytest

from furiosa.registry import Artifact, Model, ModelMetadata, Publication


@pytest.fixture(scope="module")
def tflite_artifact() -> Artifact:
    return Artifact(
        name="mlcommons_resnet50_v1.5_int8",
        family="ResNet",
        location="models/MNISTnet_uint8_quant_without_softmax.tflite",
        format="tflite",
        metadata=ModelMetadata(
            description="ResNet50 v1.5 model for MLCommons v1.1",
            publication=Publication(
                url="https://arxiv.org/abs/1512.03385.pdf",
            ),
        ),
    )


@pytest.fixture(scope="module")
def onnx_artifact() -> Artifact:
    return Artifact(
        name="mlcommons_ssd_mobilenet_v1_int8",
        family="MobileNetV1",
        location="https://github.com/furiosa-ai/furiosa-models/raw/main/models/mlcommons/mlcommons_ssd_mobilenet_v1_int8.onnx",  # noqa: E501
        format="onnx",
        metadata=ModelMetadata(
            description="MobileNet v1 model for MLCommons v1.1",
            publication=Publication(url="https://arxiv.org/abs/1704.04861.pdf"),
        ),
    )


@pytest.fixture(scope="module")
def code_artifact() -> Artifact:
    return Artifact(
        name="mlcommons_ssd_resnet34_int8",
        family="ResNet",
        location="models/model.py",
        format="code",
        metadata=ModelMetadata(
            description="ResNet34 model for MLCommons v1.1",
            publication=Publication(
                url="https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection"  # noqa: E501
            ),
        ),
    )


@pytest.fixture(scope="module")
def artifacts(tflite_artifact, onnx_artifact, code_artifact) -> List[Artifact]:
    return [tflite_artifact, onnx_artifact, code_artifact]


@pytest.fixture(scope="module")
def model_file() -> str:
    return "./tests/fixtures/models/MNISTnet_uint8_quant_without_softmax.tflite"


@pytest.fixture(scope="module")
def model_binary(model_file) -> bytes:
    with open(model_file, "rb") as data:
        return data.read()


@pytest.fixture(scope="module")
def tflite_model(model_binary) -> Model:
    return Model(
        name="mlcommons_resnet50_v1.5_int8",
        model=model_binary,
        version="v1.1",
        description="ResNet50 v1.5 model for MLCommons v1.1",
    )


@pytest.fixture(scope="module")
def onnx_model(model_binary) -> Model:
    return Model(
        name="mlcommons_ssd_mobilenet_v1_int8",
        model=model_binary,
        version="v1.1",
        description="MobileNet v1 model for MLCommons v1.1",
    )


@pytest.fixture(scope="module")
def code_model(model_binary) -> Model:
    return Model(
        name="mlcommons_ssd_resnet34_int8",
        model=model_binary,
        version="v1.1",
        description="ResNet34 model for MLCommons v1.1",
    )
