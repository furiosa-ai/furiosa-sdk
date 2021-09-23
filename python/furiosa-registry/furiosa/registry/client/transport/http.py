import pathlib
from typing import Dict, List

import aiohttp

from ... import Artifact
from .base import Loader, Transport


class HTTPTransport(Transport):
    """
    Transport for a file fetched via HTTP.

    This transport check specified URI has valid http scheme (e.g. https://, http://)
    """

    def __init__(self, headers: Dict = {}, params: Dict = {}):
        self._headers = headers
        self._params = params

    @staticmethod
    def is_supported(uri: str) -> bool:
        return any(uri.startswith(scheme) for scheme in ("https://", "http://"))

    async def fetch(self, uri: str) -> List[Artifact]:
        data = await self.download(uri)
        return Loader.load(pathlib.Path(uri).suffix[1:], data.decode())

    async def download(self, uri: str) -> bytes:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(uri, params=self._params) as response:
                data = await response.read()
                response.raise_for_status()
                return data
