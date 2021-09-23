import io
import pathlib
from typing import List, Tuple
from urllib.parse import urlparse

from ...artifact import Artifact
from .base import Loader, Transport


class S3Transport(Transport):
    """
    Transport for a file fetched via S3.

    This transport check specified URI has valid S3 scheme (e.g. s3://).
    """

    @staticmethod
    def is_supported(uri: str) -> bool:
        return uri.startswith("s3://")

    @staticmethod
    def parse(uri: str) -> Tuple[str, str]:
        """
        Parse URI to network location and path
        """
        parsed = urlparse(uri)
        return parsed.netloc, parsed.path[1:]

    async def fetch(self, uri: str) -> List[Artifact]:
        data = await self.download(uri)
        return Loader.load(pathlib.Path(uri).suffix[1:], data.decode())

    async def download(self, uri: str) -> bytes:
        import aioboto3

        async with aioboto3.Session().resource("s3") as resource:
            bucket, key = self.parse(uri)

            with io.BytesIO() as data:
                await (await resource.Object(bucket, key)).download_fileobj(data)
                return data.getvalue()
