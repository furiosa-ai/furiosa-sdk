from abc import ABC, abstractmethod
from typing import Any

from ...artifact import Artifact
from ...model import Model
from ..transport import is_relative, read, supported


class Resolver(ABC):
    """Base resolver class to resolve a artfiact to a model."""

    @abstractmethod
    async def resolve(
        self, uri: str, artifact: Artifact, version: str = "", *args: Any, **kwargs: Any
    ) -> Model:
        """Resolve a model from a artifact."""
        ...

    @staticmethod
    async def read(uri: str, path: str) -> bytes:
        """Read a file specified URI and path considerding the path."""
        if is_relative(path):
            return await read(uri, path)

        # If the path is not relative, use transport directly read the path.
        with supported(path) as transport:
            return await transport.read(path)
