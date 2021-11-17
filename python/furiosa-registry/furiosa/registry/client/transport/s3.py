import io
from typing import Tuple
from urllib.parse import urlparse

from multipledispatch import dispatch

from .base import Transport


class S3Transport(Transport):
    """Transport for S3.

    This transport check specified URI has valid S3 scheme (e.g. s3://).
    """

    scheme = "s3://"

    @staticmethod
    def is_supported(uri: str) -> bool:
        return uri.startswith(S3Transport.scheme)

    @dispatch(str, str)
    async def read(self, uri: str, path: str) -> bytes:
        assert self.is_supported(uri)
        return await self.read(f"{uri}/{path}")

    @dispatch(str)  # type: ignore
    async def read(self, location: str) -> bytes:  # noqa: F811
        import aioboto3

        async with aioboto3.Session().resource("s3") as resource:
            bucket, key = self.parse(location)

            with io.BytesIO() as data:
                await (await resource.Object(bucket, key)).download_fileobj(data)
                return data.getvalue()

    async def download(self, uri: str) -> str:
        raise NotImplementedError("S3Transport download() not yet supported")

    @staticmethod
    def parse(uri: str) -> Tuple[str, str]:
        """Parse URI to network location and path."""
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path[1:]
