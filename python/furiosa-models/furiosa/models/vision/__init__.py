import functools

from furiosa.common.thread import synchronous
from furiosa.registry import Model

# Where async models reside
from ..nonblocking import vision

__all__ = []


# Iterate over non-blocking versions of Model classes (that of ..nonblocking.vision)
# TODO: Need more precise model checking logic (as-is: is it functools.partial?)
for model in [
    getattr(vision, m) for m in dir(vision) if isinstance(getattr(vision, m), functools.partial)
]:
    # Get original function name through `functools.partial`'s metadata
    name = model.keywords['name']

    # Export synchronous version of Model class in this module scope
    globals()[name] = synchronous(model)
    __all__.append(name)


# Clean up unnecessary variables in this module
del Model, vision, functools, model, name, synchronous
