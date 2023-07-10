import io
from typing import Dict

import aiohttp
from multipledispatch import dispatch
from tqdm.asyncio import tqdm

from .base import Transport


class HTTPTransport(Transport):
    """Transport for HTTP.

    This transport check specified URI has valid http scheme (e.g. https://, http://)
    """

    schemes = ("https://", "http://")

    def __init__(self, headers: Dict = {}, params: Dict = {}):
        self._headers = headers
        self._params = params

    @staticmethod
    def is_supported(uri: str) -> bool:
        return any(uri.startswith(scheme) for scheme in HTTPTransport.schemes)

    @dispatch(str, str)
    async def read(self, uri: str, path: str) -> bytes:
        assert self.is_supported(uri)

        return await self.read(f"{uri}/{path}")

    @dispatch(str)  # type: ignore
    async def read(self, location: str) -> bytes:  # noqa: F811
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(location, params=self._params) as response:
                with io.BytesIO() as data:
                    response.raise_for_status()

                    total = int(response.headers.get("content-length", 0))

                    async for chunk in tqdm(
                        response.content.iter_chunked(1024), total=(total + 1024) // 1024
                    ):
                        data.write(chunk)

                    return data.getvalue()

    async def download(self, uri: str) -> str:
        raise NotImplementedError("HTTPTransport download() not yet supported")
