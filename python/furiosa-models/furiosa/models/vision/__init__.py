from functools import partial
from typing import Optional

import furiosa.registry as registry
from furiosa.registry import Model

version = "v1.1"
repository = "https://github.com/furiosa-ai/furiosa-artifacts"


async def load(name, *args, **kwargs) -> Optional[Model]:
    return await registry.load(uri=repository, version=version, name=name)


MLCommonsResNet50 = partial(load, name="mlcommons_resnet50")

MLCommonsMobileNet = partial(load, name="mlcommons_ssd_mobilenet")

MLCommonsSSDResNet34 = partial(load, name="mlcommons_ssd_resnet34")
