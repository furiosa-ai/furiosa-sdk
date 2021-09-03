import os
import pathlib
from typing import List

import aiofiles

from ...artifact import Artifact
from .base import Serialize, Transport


class FileTransport(Transport):
    """
    Transport for local file path.

    This resolver check specified URI is valid local file path which is existent.
    """

    @staticmethod
    def is_supported(uri: str) -> bool:
        return os.path.exists(uri)

    async def fetch(self, uri: str) -> List[Artifact]:
        return Serialize.load(
            pathlib.Path(uri).suffix[1:], (await self.download(uri)).decode()
        )

    async def download(self, uri: str) -> bytes:
        async with aiofiles.open(uri, "rb") as file:
            return await file.read()
