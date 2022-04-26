from functools import partial
from typing import Optional

from furiosa.common.thread import synchronous
import furiosa.registry as registry
from furiosa.registry import Model

__all__ = []


# Main repository to load models
repository = "https://github.com/furiosa-ai/furiosa-artifacts:v0.0.1"


async def load(name, *args, **kwargs) -> Optional[Model]:
    return await registry.load(uri=repository, name=name)


for name in synchronous(registry.list)(repository):
    model = partial(load, name=name)

    # Export Model class in this module scope
    globals()[name] = model
    __all__.append(name)


# Clean up unncessary variables in this module
del Model, Optional, load, model, name, partial, synchronous
