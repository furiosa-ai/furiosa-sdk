"""FuriosaAI registry client"""

from typing import List

from ..errors import URINotFound
from ..model import Model
from .resolver import resolve
from .transport import fetch

__all__ = [
    "request",
]

descriptors = ["artifact.toml", "artifact.yaml"]


async def request(uri: str, version: str = "") -> List[Model]:
    """
    Request models from the specified registry

    Args:
        uri (str): Registry URI which have a descriptor file (artifact.toml).

    Returns:
        List[Model]: Models loaded from the registry.
    """

    for descriptor in descriptors:
        try:
            artifacts = await fetch(f"{uri}/{descriptor}")

            assert type(artifacts) == list

            return [await resolve(artifact, version) for artifact in artifacts]
        except URINotFound:
            # Try other descriptor file (e.g. artifact.yaml)
            pass

    raise URINotFound(
        " and ".join(f"{uri}/{descriptor}" for descriptor in descriptors)
    )
