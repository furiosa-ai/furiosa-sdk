from functools import partial
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastapi import FastAPI
import numpy as np

from furiosa.registry import TransportNotFound, transport
from furiosa.server.model import Model
from furiosa.server.registry import InMemoryRegistry
from furiosa.server.repository import Repository

from .apps.repository import repository
from .model import CPUServeModel, NPUServeModel, OpenVINOServeModel, ServeModel
from .telemetry import setup_metrics, setup_otlp


class ServeAPI:
    def __init__(
        self,
        repository: Repository = repository,
        **kwargs: Any,
    ):
        self._app = FastAPI(**kwargs, on_startup=[self.load, self.setup_telemetry])
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

    def model(self, device: str) -> Callable[[Any], Awaitable[ServeModel]]:
        def register(model: ServeModel):
            # Register model config for model discovery
            assert self._registry is not None
            self._registry.register(model.config)

            # Keep inner model config for model discovery
            self._models[model.inner] = model

        constructors = {
            "nux": partial(nux, app=self.app, on_create=register),
            "python": partial(python, app=self.app, on_create=register),
            "openvino": partial(openvino, app=self.app, on_create=register),
        }

        try:
            # https://github.com/python/mypy/issues/10740
            return constructors[device]  # type: ignore
        except KeyError:
            raise ValueError(
                f"Invalid device {device} for ServeModel. Available devices: {constructors.keys()}"
            )

    async def load(self):
        for inner, serve_model in self._models.items():
            # Load models via repository
            await self._repository.load(inner)

    def setup_telemetry(self):
        app_name = os.environ.get("FURIOSA_SERVING_APP_NAME", "furiosa-serving")
        otlp_endpoint = os.environ.get("FURIOSA_SERVING_OTLP_ENDPOINT")
        setup_metrics(self._app, app_name)
        setup_otlp(self._app, app_name, otlp_endpoint)

    def _on_load(self, model: Model):
        """Apply callback function which expose API endpoints when model is loaded."""
        for inner, serve_model in self._models.items():
            # (name, version) pair gurantees Model identity
            # TODO(yan): Refactor repository API to load ServeModel directly
            if inner.name == model.name and inner.version == model.version:
                serve_model.expose()

    def _on_unload(self, inner: Model):
        """Apply callback function which hide API endpoints when model is unloaded."""
        serve_model = self._models[inner]
        serve_model.hide()


def fallback(location: str) -> str:
    # Add file prefix if the scheme is not discoverable
    try:
        with transport.supported(location):
            return location
    except TransportNotFound:
        return "file://" + location


async def python(
    name: str,
    predict: Callable[[Any, Any], Awaitable[Any]],
    *,
    app: FastAPI,
    on_create: Callable[[ServeModel], None],
    version: Optional[str] = None,
    description: Optional[str] = None,
    preprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
    postprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
) -> CPUServeModel:
    model = CPUServeModel(
        app,
        name,
        version=version,
        description=description,
        predict=predict,
        preprocess=preprocess,
        postprocess=postprocess,
    )

    on_create(model)
    return model


async def nux(
    name: str,
    location: str,
    *,
    blocking: bool = True,
    app: FastAPI,
    on_create: Callable[[ServeModel], None],
    version: Optional[str] = None,
    description: Optional[str] = None,
    preprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
    postprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
    npu_device: Optional[str] = None,
    batch_size: Optional[int] = None,
    worker_num: Optional[int] = None,
    compiler_config: Optional[Dict] = None,
) -> NPUServeModel:
    model = NPUServeModel(
        app,
        name,
        blocking=blocking,
        model=await transport.read(fallback(location)),
        version=version,
        description=description,
        preprocess=preprocess,
        postprocess=postprocess,
        npu_device=npu_device,
        batch_size=batch_size,
        worker_num=worker_num,
        compiler_config=compiler_config,
    )

    on_create(model)
    return model


async def openvino(
    name: str,
    location: str,
    *,
    app: FastAPI,
    on_create: Callable[[ServeModel], None],
    version: Optional[str] = None,
    description: Optional[str] = None,
    compiler_config: Optional[Dict] = None,
    preprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
    postprocess: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
) -> OpenVINOServeModel:
    model = OpenVINOServeModel(
        app,
        name,
        model=await transport.read(fallback(location)),
        version=version,
        description=description,
        compiler_config=compiler_config,
        preprocess=preprocess,
        postprocess=postprocess,
    )

    on_create(model)
    return model
