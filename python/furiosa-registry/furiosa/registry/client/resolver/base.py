from abc import ABC, abstractmethod

from ...artifact import Artifact
from ...model import Model


class Resolver(ABC):
    """
    Base resolver class to resolve a artfiact to a model.
    """

    @abstractmethod
    async def resolve(self, artifact: Artifact, version: str = "") -> Model:
        """
        Resolve a model from a artifact
        """
        ...
