from ...artifact import Artifact, Format
from ...model import Model
from .code import CodeResolver
from .serial import SerialResolver

__all__ = [
    "CodeResolver",
    "SerialResolver",
]

resolvers = {
    Format.Code: CodeResolver(),
    Format.TFLite: SerialResolver(),
    Format.ONNX: SerialResolver(),
}


async def resolve(artifact: Artifact, version: str = "") -> Model:
    return await resolvers[artifact.format].resolve(artifact, version)
