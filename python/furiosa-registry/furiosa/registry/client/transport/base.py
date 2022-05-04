from abc import ABC, abstractmethod
import os

from multipledispatch import dispatch


class Transport(ABC):
    """Base transport class to read/download from various registry sources."""

    @staticmethod
    def is_supported(uri: str) -> bool:
        """Decide whether this transport supports the specified URI.

        Args:
            uri (str): Registry URI to locate the models.

        Returns:
            bool: This transport supports the URI or not.
        """
        raise NotImplementedError

    @dispatch(str, str)
    async def read(self, uri: str, path: str) -> bytes:
        """Read a file binary data from the specified registry URI and path.

        This is a high level function to use `read(location)` internally.

        Args:
            uri (str): Registry URI to locate the models.
            path (str): Relative file path in the repositry to read.

        Returns:
            bytes: Downloaded binary data.
        """
        raise NotImplementedError

    @dispatch(str)  # type: ignore
    async def read(self, location: str) -> bytes:  # noqa: F811
        """Read a file binary data from the specified location.

        Args:
            location (str): Location(URL) to read the file. This should be valid URL to download.

        Returns:
            bytes: Downloaded binary data.
        """
        raise NotImplementedError

    @abstractmethod
    async def download(self, uri: str) -> str:
        """Download a registry directory into local destination.

        Args:
            uri (str): Registry URI to download the data.

        Returns:
            str: Destination directory name. This directory will be located in `cache` directory.
        """
        raise NotImplementedError

    @property
    def cache_directory(self) -> str:
        """Cache directory to save downloaded files."""
        return os.path.expanduser(
            os.getenv(
                "FURIOSA_REGISTRY_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "furiosa"),
            )
        )
