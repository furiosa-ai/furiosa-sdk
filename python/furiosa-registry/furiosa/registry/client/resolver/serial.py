from ...artifact import Artifact
from ...model import Model
from ..transport import download
from .base import Resolver


class SerialResolver(Resolver):
    """
    Model resolver from a artifact with a serialized(protobuf, flatbuffer) data
    """

    async def resolve(self, artifact: Artifact, version: str = "") -> Model:
        # FIXME(yan): Nothing interesting here now. What else we provide for richer API?
        model = await download(artifact.location)

        return Model(
            name=artifact.name,
            description=artifact.description,
            config=artifact.config,

            model=model,
            version=version,
        )
