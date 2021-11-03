from ...artifact import Artifact, Format
from ...model import Model
from .code import CodeResolver
from .serial import SerialResolver

__all__ = ["CodeResolver", "SerialResolver", "resolve"]

resolvers = {
    Format.Code: CodeResolver(),
    Format.TFLite: SerialResolver(),
    Format.ONNX: SerialResolver(),
}


async def resolve(uri: str, artifact: Artifact, *args, **kwargs) -> Model:
    """Resolve Artifact from the source to Model with the version.

    Args:
        uri (str): Registry source URI which have the Artifact.
        artifact (Artifact): Artifact to be resolved.
        args, kwargs (Any): Arguments for Model instantiation.

    Returns:
        Model: A Model created from the Artifact.
    """
    return await resolvers[artifact.format].resolve(uri, artifact, *args, **kwargs)
