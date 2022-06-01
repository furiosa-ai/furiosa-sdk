from functools import partial
from typing import Optional

from furiosa.common.thread import synchronous
import furiosa.registry as registry
from furiosa.registry import Model

__all__ = []


async def load(name, *args, **kwargs) -> Optional[Model]:
    # Import registry again to avoid exporting non-Model variable
    import furiosa.registry

    return await furiosa.registry.load(
        uri="https://github.com/furiosa-ai/furiosa-artifacts:v0.0.1", name=name, *args, **kwargs
    )


for name in synchronous(registry.list)("https://github.com/furiosa-ai/furiosa-artifacts:v0.0.1"):
    model = partial(load, name=name)

    # Export Model class in this module scope
    globals()[name] = model
    __all__.append(name)


# Clean up unncessary variables in this module
del Model, Optional, load, model, name, partial, synchronous, registry
