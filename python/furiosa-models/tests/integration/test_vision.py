import pytest

from furiosa.models.vision import (
    EfficientNetV2_S,
    MLCommonsResNet50,
    MLCommonsSSDMobileNet,
    MLCommonsSSDResNet34,
)
from furiosa.registry import Model


@pytest.mark.asyncio
async def test_resnet50():
    model: Model = await MLCommonsResNet50()
    assert model.name == "MLCommonsResNet50"


@pytest.mark.asyncio
async def test_mobilenet():
    model: Model = await MLCommonsSSDMobileNet()
    assert model.name == "MLCommonsSSDMobileNet"


@pytest.mark.asyncio
async def test_resnet34():
    model: Model = await MLCommonsSSDResNet34()
    assert model.name == "MLCommonsSSDResNet34"


@pytest.mark.asyncio
async def test_efficientnet():
    model: Model = await EfficientNetV2_S()
    assert model.name == "EfficientNetV2_S"
