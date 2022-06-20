from furiosa.models.vision import (
    EfficientNetV2_S,
    MLCommonsResNet50,
    MLCommonsSSDMobileNet,
    MLCommonsSSDResNet34,
)
from furiosa.registry import Model


def test_resnet50():
    model: Model = MLCommonsResNet50()
    assert model.name == "MLCommonsResNet50"


def test_mobilenet():
    model: Model = MLCommonsSSDMobileNet()
    assert model.name == "MLCommonsSSDMobileNet"


def test_resnet34():
    model: Model = MLCommonsSSDResNet34()
    assert model.name == "MLCommonsSSDResNet34"


def test_efficientnet():
    model: Model = EfficientNetV2_S()
    assert model.name == "EfficientNetV2_S"
