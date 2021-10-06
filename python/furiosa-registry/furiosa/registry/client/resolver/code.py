from ...artifact import Artifact
from ...model import Model
from .base import Resolver


class CodeResolver(Resolver):
    """
    Model resolver from a artifact with a code (Pytorch Module, TensorFlow SavedModel)
    """

    async def resolve(self, artifact: Artifact, version: str = "") -> Model:
        raise NotImplementedError("CodeResolver is not yet implemented")
