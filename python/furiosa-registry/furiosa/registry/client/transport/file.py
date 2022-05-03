import os
import pathlib
import shutil

import aiofiles
from multipledispatch import dispatch

from ...utils import removeprefix
from .base import Transport


class FileTransport(Transport):
    """Transport for local file path.

    This transport check specified URI has valid local file scheme (e.g. file://).
    """

    scheme = "file://"

    @staticmethod
    def is_supported(uri: str) -> bool:
        return uri.startswith(FileTransport.scheme)

    @dispatch(str, str)
    async def read(self, uri: str, path: str) -> bytes:
        assert self.is_supported(uri)

        return await self.read(os.path.join(removeprefix(uri, self.scheme), path))

    @dispatch(str)  # type: ignore
    async def read(self, location: str) -> bytes:  # noqa: F811
        file = removeprefix(location, self.scheme)
        async with aiofiles.open(file, "rb") as f:
            return await f.read()

    async def download(self, uri: str) -> str:
        """Download a registry directory into local destination.

        FileTransport download is just to copy local directory into the cache.
        """
        # TODO(ileixe): Implement cache layer.
        assert self.is_supported(uri)

        directory = self.cache_directory
        if not os.path.exists(directory):
            os.mkdir(directory)

        location = removeprefix(uri, self.scheme)

        src = location
        dst = os.path.join(directory, pathlib.Path(location).absolute().name)

        shutil.rmtree(dst, ignore_errors=True)
        return shutil.copytree(src, dst)
