"""FuriosaAI registry client."""

import asyncio
import inspect
import logging
from typing import Any, List, Optional

from ..model import Model
from ..utils import import_module, python_path, working_directory
from .transport import download

__all__ = ["list", "load", "help"]


Logger = logging.getLogger(__name__)

# Default descriptor where Model classes reside
Descriptor = "artifacts.py"


async def load(uri: str, name: str, *args: Any, **kwargs: Any) -> Optional[Model]:
    """Load models from the specified registry URI.

    Args:
        uri (str): Registry URI which have a descriptor file (artifacts.py).
        name (str): Model name in a descriptor file.
        args, kwargs (Any): Arguments for Model instantiation.

    Returns:
        Model: A model loaded from the registry.

    Raises:
        ModuleNotFoundError: If descriptor file not found in the registry.
    """

    directory = await download(uri)

    module = import_module(directory, Descriptor)

    if name not in dir(module):
        Logger.debug(f"{name} not found in {module}")
        return None

    entry = getattr(module, name)

    # Replace working directory to use file system dependent function like 'open' in a registry.
    with python_path(directory), working_directory(directory):
        if asyncio.iscoroutinefunction(entry):
            return await entry(*args, **kwargs)

        return entry(*args, **kwargs)


async def list(uri: str) -> List[str]:
    """List Artifacts from the specified registry URI.

    Args:
        uri (str): Registry URI which have a descriptor file (artifacts.py).

    Returns:
        List[str]: Available model names in the registry

    Raises:
        ModuleNotFoundError: If descriptor file not found in the registry.
    """

    def is_model(value):
        return inspect.isfunction(value) and issubclass(
            inspect.signature(value).return_annotation, Model
        )

    module = import_module(await download(uri), Descriptor)
    return [name for name in dir(module) if is_model(getattr(module, name))]


async def help(uri: str, name: str, version: Optional[str] = None) -> str:
    """Show the documentation for the model

    Args:
        uri (str): Registry URI which have a descriptor file (artifacts.py).
        name (str): Model name in a descriptor file.
        version (Optional[str]): Model version to be created.

    Returns:
        str: Documentation string for the model.

    Raises:
        ModuleNotFoundError: If descriptor file not found in the registry.
    """

    return (await load(uri, name, version)).__doc__
