"""FuriosaAI registry transport"""

from functools import partial
from typing import List, Union

from ...artifact import Artifact
from ...errors import TransportNotFound
from .base import Transport
from .file import FileTransport
from .http import HTTPTransport
from .s3 import S3Transport

__all__ = [
    "FileTransport",
    "HTTpTransport",
    "S3Transport",
    "Transport",
]

transports = [
    FileTransport(),
    HTTPTransport(),
    S3Transport(),
]


async def request(uri: str, method: str) -> Union[bytes, List[Artifact]]:
    for transport in transports:
        if transport.is_supported(uri):
            return await getattr(transport, method)(uri)

    raise TransportNotFound(uri, transports)


fetch = partial(request, method="fetch")
download = partial(request, method="download")
