from typing import Callable, Dict, List, Optional, Union

from fastapi import FastAPI
from fastapi.routing import Mount
import numpy as np

from furiosa.runtime.tensor import TensorDesc
from furiosa.server import ModelConfig, NuxModel


class ServeModel:
    def __init__(
        self,
        app: FastAPI,
        name: str,
        *,
        model: Union[str, bytes],
        version: Optional[str] = None,
        description: Optional[str] = None,
        npu_device: Optional[str] = None,
        compiler_config: Optional[Dict] = None,
    ):
        self._app = app
        self._config = ModelConfig(
            name=name,
            model=model,
            version=version,
            description=description,
            npu_device=npu_device,
            compiler_config=compiler_config,
        )

        self._model = NuxModel(self._config)
        self._routes: Dict[Callable, Callable] = {}

    def expose(self):
        """
        Expose FastAPI route API endpoint
        """
        for func, decorator in self._routes.items():
            # Decorate the path operation function to expose endpoint
            decorator(func)

    def hide(self):
        """
        Hide FastAPI route API endpoint
        """
        # Gather routes not in sub applications
        routes = [route for route in self._app.routes if not isinstance(route, Mount)]

        # Target routes to be removed
        targets = [route for route in routes if route.endpoint in self._routes]  # type: ignore

        # Unregister path operation functions to hide endpoint
        for route in targets:
            self._app.routes.remove(route)

    async def predict(self, payload: List[np.ndarray]) -> List[np.ndarray]:
        return await self._model.predict(payload)

    @property
    def inner(self) -> NuxModel:
        return self._model

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def inputs(self) -> List[TensorDesc]:
        return self._model.session.inputs()

    @property
    def outputs(self) -> List[TensorDesc]:
        return self._model.session.outputs()

    def _method(self, kind: str, *args, **kwargs) -> Callable:
        def decorator(func):
            """
            Register FastAPI path operation function to be used later.

            The function will be registerd into FastAPI app when model is loaded.
            """
            self._routes[func] = getattr(self._app, kind)(*args, **kwargs)
            return func

        return decorator

    def get(self, *args, **kwargs) -> Callable:
        return self._method("get", *args, **kwargs)

    def put(self, *args, **kwargs) -> Callable:
        return self._method("put", *args, **kwargs)

    def post(self, *args, **kwargs) -> Callable:
        return self._method("post", *args, **kwargs)

    def delete(self, *args, **kwargs) -> Callable:
        return self._method("delete", *args, **kwargs)

    def head(self, *args, **kwargs) -> Callable:
        return self._method("head", *args, **kwargs)

    def patch(self, *args, **kwargs) -> Callable:
        return self._method("patch", *args, **kwargs)

    def trace(self, *args, **kwargs) -> Callable:
        return self._method("trace", *args, **kwargs)
