from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import toml
import yaml

from ...artifact import Artifact


class Transport(ABC):
    """
    Base transport class to fetch/download from various sources.
    """

    @staticmethod
    def is_supported(uri: str) -> bool:
        """
        Decicde wether this transport support specified URI.
        """
        ...

    @abstractmethod
    async def fetch(self, uri: str) -> List[Artifact]:
        """
        Fetch artifact config from the specified URI.
        """
        ...

    @abstractmethod
    async def download(self, uri: str) -> bytes:
        """
        Download binary data from the specified URI.
        """
        ...


class Serialize:
    """
    Serialize configuration file string into Artifacts.
    """

    loaders: Dict[str, Callable] = {
        "yaml": yaml.safe_load,
        "toml": toml.loads,
    }

    @classmethod
    def load(cls, format: str, data: str) -> List[Artifact]:
        serialized: Dict = cls.loaders[format](data)

        return [Artifact.parse_obj(artifact) for artifact in serialized["artifacts"]]
