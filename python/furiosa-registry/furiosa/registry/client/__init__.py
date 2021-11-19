"""FuriosaAI registry client."""

from typing import Any, Callable, Dict, List, Optional

import toml
import yaml

from ..artifact import Artifact
from ..model import Model
from .resolver import resolve
from .transport import read

__all__ = ["list", "load", "help"]


async def load(
    uri: str, name: str, version: Optional[str] = None, *args: Any, **kwargs: Any
) -> Optional[Model]:
    """Load models from the specified registry URI.

    Args:
        uri (str): Registry URI which have a descriptor file (artifact.toml).
        name (str): Model name in a descriptor file.
        version (Optional[str]): Model version in a descriptor file.
        args, kwargs (Any): Arguments for Model instantiation.

    Returns:
        Model: A model loaded from the registry.
    """

    artifacts = [
        artifact
        for artifact in await list(uri)
        if artifact.name == name and artifact.version == version
    ]

    if not artifacts:
        return None

    assert len(artifacts) == 1, "Model name should be unique in a artifact descriptor"

    return await resolve(uri, artifacts[0], *args, **kwargs)


async def list(uri: str) -> List[Artifact]:
    """List Artifacts from the specified registry URI.

    Args:
        uri (str): Registry URI which have a descriptor file (artifact.toml).

    Returns:
        List[Artifact]: Artifacts from the registry.
    """

    def serialize(format: str, data: str) -> List[Artifact]:
        """Serialize artifact from data specified."""
        loaders: Dict[str, Callable] = {
            "yaml": yaml.safe_load,
            "toml": toml.loads,
        }

        serialized: Dict = loaders[format](data)
        return [Artifact.parse_obj(artifact) for artifact in serialized["artifacts"]]

    # TODO(ileixe): Implement fallback for different descriptors like artifact.toml.
    descriptor = "artifacts.yaml"

    data = (await read(uri, descriptor)).decode()
    return serialize(descriptor.split(".")[1], data)


async def help(uri: str, name: str, version: str = "") -> str:
    """Show the documentation for the model

    Args:
        uri (str): Registry URI which have a descriptor file (artifact.toml).
        name (str): Model name in a descriptor file.
        version (Optional[str]): Model version to be created.

    Returns:
        str: Documentation string for the model.
    """
    return (await load(uri, name, version)).__doc__
