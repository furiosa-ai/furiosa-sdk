from abc import ABC, abstractmethod
import glob
import os
from typing import List

import yaml

from .errors import ModelNotFound
from .settings import FileRegistryConfig, ModelConfig


class Registry(ABC):
    @abstractmethod
    async def list(self) -> List[ModelConfig]:
        """
        Get the (loaded + unloaded) model configs
        """
        pass

    async def find(self, name: str) -> ModelConfig:
        """
        Find a (loaded + unloaded) model configs
        """
        # TODO(yan): Choose specific version instead of first item
        first = next(
            (config for config in await self.list() if config.name == name),
            None,
        )

        if first is None:
            raise ModelNotFound(name)

        return first


class InMemoryRegistry(Registry):
    def __init__(self, model_configs: List[ModelConfig] = []):
        self._model_configs = model_configs

    async def list(self) -> List[ModelConfig]:
        """
        Get the model configs in-memory
        """
        return self._model_configs

    def register(self, model_config: ModelConfig) -> bool:
        """
        Register a model config in this in-memory registry

        Returns:
            bool: True for success, False if the config is already existed
        """
        if model_config in self._model_configs:
            return False

        self._model_configs.append(model_config)
        return True

    def unregister(self, model_config: ModelConfig) -> bool:
        """
        Unregister a model config from this in-memory registry

        Returns:
            bool: True for success, False if the config is not existed
        """
        if model_config not in self._model_configs:
            return False

        self._model_configs.remove(model_config)
        return True


class FileRegistry(Registry):
    def __init__(self, config: FileRegistryConfig):
        self._config = config

    async def list(self) -> List[ModelConfig]:
        """
        Get the model configs from file with yaml suffix in the specified directory
        """
        pattern = os.path.join(self._config.repository_root, "*.yaml")
        matches = glob.glob(pattern, recursive=True)

        def load(config_file):
            models = yaml.safe_load(config_file)["model_config_list"]
            return [ModelConfig.parse_obj(model) for model in models]

        return sum((load(config) for config in matches), [])
