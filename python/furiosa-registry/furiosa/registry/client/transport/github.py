import io
import os
import shutil
from typing import Tuple
import zipfile

from multipledispatch import dispatch

from ...utils import removeprefix
from .http import HTTPTransport


class GithubTransport(HTTPTransport):
    """Transport for Github repository.

    This transport check specified URI has valid Github repository URL. (e.g. https://github.com/)
    """

    schemes = ("https://github.com/", "http://github.com/")

    @staticmethod
    def is_supported(uri: str) -> bool:
        return any(uri.startswith(prefix) for prefix in GithubTransport.schemes)

    @dispatch(str, str)
    async def read(self, uri: str, path: str) -> bytes:
        """Read a file binary data from the specified registry URI and path.

        Note that Github downloadable URL is different from the URI itself. So We are replacing the
        URI and path into the valid URL and calling `read(location)` of `HTTPTransport` here.
        """
        assert self.is_supported(uri)

        owner, repo, branch = self.parse(uri)
        return await self.read(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")

    @dispatch(str)  # type: ignore
    async def read(self, location: str) -> bytes:  # noqa: F811
        return await super().read(location)

    def parse(self, uri: str) -> Tuple[str, str, str]:
        """Parse URI following Github repository scheme.

        e.g. https://github.com/furiosa-ai/furiosa-models:main to (furiosa-ai, furiosa-models, main)
        """

        # Remove Github URI prefix
        for prefix in self.schemes:
            uri = removeprefix(uri, prefix)

        # Default branch is "main"
        branch = "main"
        owner, repo = uri.split("/")
        if ":" in repo:
            repo, branch = repo.split(":")

        return owner, repo, branch

    async def download(self, uri: str):
        def unzip(data: bytes, where: str) -> str:
            with zipfile.ZipFile(io.BytesIO(data)) as f:
                name = f.infolist()[0].filename

                target = os.path.join(where, name)

                shutil.rmtree(target, ignore_errors=True)

                f.extractall(where)
            return target

        # TODO(ileixe): Impmlement cache layer.
        assert self.is_supported(uri)

        directory = self.cache_directory
        if not os.path.exists(directory):
            os.mkdir(directory)

        owner, repo, branch = self.parse(uri)

        data = await self.read(f"https://github.com/{owner}/{repo}/archive/{branch}.zip")

        src = unzip(data, directory)
        dst = os.path.join(directory, f"{owner}_{repo}_{branch}")

        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(src, dst)

        return dst
