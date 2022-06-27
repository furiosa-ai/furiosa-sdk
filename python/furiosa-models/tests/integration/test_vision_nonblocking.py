import pytest

from furiosa.models.nonblocking.vision import (
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
