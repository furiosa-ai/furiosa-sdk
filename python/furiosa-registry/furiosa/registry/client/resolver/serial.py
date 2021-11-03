from ...artifact import Artifact
from ...model import Model
from .base import Resolver


class SerialResolver(Resolver):
    """Model resolver from a artifact with a serialized(protobuf, flatbuffer) data."""

    async def resolve(self, uri: str, artifact: Artifact, *args, **kwargs) -> Model:
        model = Model(
            name=artifact.name,
            description=artifact.metadata and artifact.metadata.description,
            model=await self.read(uri, artifact.location),
            version=artifact.version,
        )

        if artifact.doc:
            model.__doc__ = (await self.read(uri, artifact.doc)).decode()

        return model
