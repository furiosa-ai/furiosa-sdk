"""FuriosaAI registry transport"""

from contextlib import contextmanager
from typing import Iterator

from multipledispatch import dispatch

from ...errors import TransportNotFound
from .base import Transport
from .file import FileTransport
from .github import GithubTransport
from .http import HTTPTransport
from .s3 import S3Transport

__all__ = [
    "transports",
    "FileTransport",
    "GithubTransport",
    "HTTPTransport",
    "S3Transport",
    "Transport",
    "supported",
    "read",
    "download",
    "is_relative",
]

transports = [FileTransport(), GithubTransport(), HTTPTransport(), S3Transport()]


@contextmanager
def supported(uri: str) -> Iterator[Transport]:
    """Supported transport for the URI."""
    for transport in transports:
        if transport.is_supported(uri):
            # Note that each transport should be exclusive as we are returning first one.
            # TODO(ileixe): Add more logging to clarify what's going on here to user.
            yield transport
            return

    raise TransportNotFound(uri, transports)


@dispatch(str, str)
async def read(uri: str, path: str) -> bytes:
    """Read a file binary data from the registry URI and path with a transport which supports the URI.

    Args:
        uri (str): Registry URI to locate the models.
        path (str): Relative file path in the repositry to read.

    Returns:
        bytes: Downloaded binary data.

    Raises:
        TransportNotFound: If all the available transports are not supporting the URI.
    """
    with supported(uri) as transport:
        return await transport.read(uri, path)


@dispatch(str)
async def read(location: str) -> bytes:  # noqa: F811
    """Read a file binary data from the specified location with a transport which supports the URI.

    Args:
        location (str): Location(URL) to read the file. This should be valid URL to download.

    Returns:
        bytes: Downloaded binary data.

    Raises:
        TransportNotFound: If all of the available transports are not supporing the URI.
    """
    with supported(location) as transport:
        return await transport.read(location)


async def download(uri: str) -> str:
    """Download a registry directory into local destination with a transport which supports the URI.

    Args:
        uri (str): Registry URI to download the data.

    Returns:
        str: Destination directory name. This directory will be located in `cache` directory.

    Raises:
        TransportNotFound: If all the available transports are not supporting the URI.
    """
    with supported(uri) as transport:
        return await transport.download(uri)


def is_relative(path: str) -> bool:
    """Is this path relative path?

    If all the available transports are not supporting the path, we assume that it's a relative
    path. You should find the path from the registry URI if it's a relative path.
    """
    return all(not transport.is_supported(path) for transport in transports)
