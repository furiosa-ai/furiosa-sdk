import asyncio
from typing import Callable, Dict, List, Optional

from .errors import ModelNotFound
from .model import Model
from .registry import Registry
from .settings import ModelConfig


class Repository:
    def __init__(
        self,
        registries: List[Registry],
        on_load: Optional[Callable[[Model], None]] = None,
        on_unload: Optional[Callable[[Model], None]] = None,
    ):
        self._registries = registries
        # Ready models which are loaded. Schema: {model_name: {model_version: Model}}
        self._models: Dict[str, Dict[str, Model]] = {}

        self.on_load = on_load
        self.on_unload = on_unload

    @property
    def registries(self) -> List[Registry]:
        return self._registries

    async def load(self, model: Model) -> bool:
        """
        Load a specified model

        Register a model in the internal directory to maintain loaded models.
        """
        versions = self._models.setdefault(model.name, {})

        version = model.version or "default"

        if version in versions:
            # Do not load again when model is already ready
            return True

        # We load the specified model again even if it was already loaded
        versions[version] = model

        loaded = await model.load()
        if loaded and self.on_load:
            self.on_load(model)
        return loaded

    async def unload(self, name: str) -> bool:
        """
        Unload models with a specified name

        Unregister models in the internal directory to maintain loaded models.
        """
        if name not in self._models:
            raise ModelNotFound(name)

        versions = self._models.pop(name)
        for model in versions.values():
            await model.unload()
            if self.on_unload:
                self.on_unload(model)
        return True

    async def get_model(self, name: str, version: str = None) -> Model:
        """
        Get a specified loaded model with name and version
        """
        version = version or "default"

        if name not in self._models:
            raise ModelNotFound(name, version)

        if version not in self._models[name]:
            raise ModelNotFound(name, version)

        return self._models[name][version]

    async def get_models(self) -> List[Model]:
        """
        Get the specified loaded models
        """
        return [model for versions in self._models.values() for model in versions.values()]

    async def list(self) -> List[ModelConfig]:
        """
        Get the (loaded + unloaded) model configs from several registries
        """
        return sum(
            list(await asyncio.gather(*(registry.list() for registry in self._registries))),
            [],
        )

    async def find(self, name: str) -> ModelConfig:
        """
        Find a (loaded + unloaded) model configs from several registries
        """
        # TODO(yan): Choose specific version instead of first item
        for registry in self._registries:
            try:
                config = await registry.find(name)
                return config
            except ModelNotFound:
                pass

        raise ModelNotFound(name)
