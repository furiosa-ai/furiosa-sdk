import pytest

from furiosa.models.vision import MLCommonsMobileNet, MLCommonsResNet50, MLCommonsSSDResNet34
from furiosa.registry import Model


@pytest.mark.asyncio
async def test_resnet50():
    model: Model = await MLCommonsResNet50()
    assert model.name == "mlcommons_resnet50"


@pytest.mark.asyncio
async def test_mobilenet():
    model: Model = await MLCommonsMobileNet()
    assert model.name == "mlcommons_ssd_mobilenet"


@pytest.mark.asyncio
async def test_resnet34():
    model: Model = await MLCommonsSSDResNet34()
    assert model.name == "mlcommons_ssd_resnet34"
