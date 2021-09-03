import asyncio
from functools import partial
from typing import List

from furiosa.registry import Model, request

version = "main"
repository = f"https://raw.githubusercontent.com/furiosa-ai/furiosa-models/{version}"

# TODO(yan): Provide lazy loading interface.
models: List[Model] = asyncio.run(request(uri=repository, version=version))


def Model(name: str):
    return next(iter(model for model in models if model.name == name))


MLCommons_ResNet50_V1_5 = partial(Model, name="mlcommons_resnet50_v1.5_int8")

MLCommons_MobileNetV1 = partial(Model, name="mlcommons_ssd_mobilenet_v1_int8")

MLCommons_SSD1200_ResNet34 = partial(Model, name="mlcommons_ssd_resnet34_int8")
