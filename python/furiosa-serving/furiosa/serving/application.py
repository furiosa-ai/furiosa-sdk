import os
from typing import Any, Dict, Optional

from fastapi import FastAPI

from furiosa.registry.client import transport
from furiosa.server.model import Model
from furiosa.server.registry import InMemoryRegistry
from furiosa.server.repository import Repository

from .apps.repository import repository
from .model import ServeModel


class ServeAPI:
    def __init__(
        self,
        repository: Repository = repository,
        **kwargs: Any,
    ):
        self._app = FastAPI(**kwargs, on_startup=[self.load])
        self._models: Dict[Model, ServeModel] = {}
        self._repository = repository
        self._registry = next(
            (
                registry
                for registry in repository.registries
                if isinstance(registry, InMemoryRegistry)
            ),
            None,
        )

        assert (
            self._registry is not None
        ), "InMemoryRegistry is required to use Repository in ServeAPI"

        self._repository.on_load = self._on_load
        self._repository.on_unload = self._on_unload

    @property
    def app(self) -> FastAPI:
        return self._app

    async def model(
        self,
        name: str,
        *,
        location: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        npu_device: Optional[str] = None,
        compiler_config: Optional[Dict] = None,
    ):
        # Assume that relative path is only for file
        if transport.is_relative(location):
            location = os.path.join("file://", location)

        model = ServeModel(
            app=self._app,
            name=name,
            model=await transport.read(location),
            version=version,
            description=description,
            npu_device=npu_device,
            compiler_config=compiler_config,
        )

        self._models[model.inner] = model

        # Register model config for model discovery
        assert self._registry is not None
        self._registry.register(model.config)

        return model

    async def load(self):
        print("application load")
        for inner, serve_model in self._models.items():
            # Load models via repository
            await self._repository.load(inner)

    def _on_load(self, model: Model):
        """
        Callback function which expose API endpoints when model is loaded
        """
        for inner, serve_model in self._models.items():
            # (name, version) pair gurantees Model identity
            # TODO(yan): Refactor repository API to load ServeModel directly
            if inner.name == model.name and inner.version == model.version:
                serve_model.expose()

    def _on_unload(self, inner: Model):
        """
        Callback function which hide API endpoints when model is unloaded
        """
        serve_model = self._models[inner]
        serve_model.hide()
