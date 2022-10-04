"""Model class for prediction/explanation."""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, overload

import numpy as np

from furiosa.common.thread import asynchronous
from furiosa.runtime import session
from furiosa.runtime.tensor import TensorArray, TensorDesc

from .settings import ModelConfig, NuxModelConfig, OpenVINOModelConfig
from .types import (
    InferenceRequest,
    InferenceResponse,
    MetadataModelResponse,
    RequestInput,
    ResponseOutput,
)


class Model(ABC):
    """Base model class for every runtime."""

    def __init__(self, config: ModelConfig):
        self.ready = False
        self._config = config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def version(self) -> Optional[str]:
        return self._config.version

    async def metadata(self) -> MetadataModelResponse:
        # FIXME(yan): Model metadata is not currently provided
        return MetadataModelResponse(
            name=self.name,
            platform=self._config.platform,
            versions=[self._config.version or "default"],
            inputs=self._config.inputs,
            outputs=self._config.outputs,
        )

    async def load(self) -> bool:
        self.ready = True
        return self.ready

    async def unload(self):
        self.ready = False

    @overload
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        ...

    @overload
    async def predict(self, payload: List[np.ndarray]) -> List[np.ndarray]:
        ...

    @abstractmethod
    async def predict(self, payload):
        pass


class NuxModel(Model):
    """Model Nux runtime."""

    def __init__(self, config: NuxModelConfig):
        super().__init__(config)

    async def predict(self, payload):
        if isinstance(payload, InferenceRequest):
            # Python list to Numpy array
            inputs = [
                self.decode(tensor, request)
                for tensor, request in zip(self._session.inputs(), payload.inputs)
            ]
        else:
            inputs: List[np.ndarray] = payload

        # Infer from Nux
        tensors = [tensor.numpy() for tensor in await self.run(inputs)]

        if isinstance(payload, InferenceRequest):
            # TensorArray(Numpy array) to Python list
            outputs = [
                # TODO(yan): Support named tensor. See furiosa-ai/npu-tools#2421
                self.encode(str(i), tensor)
                for i, tensor in enumerate(tensors)
            ]

            return InferenceResponse(
                model_name=self.name, model_version=self.version, outputs=outputs
            )
        else:
            return tensors

    async def run(self, inputs: Sequence[np.ndarray]) -> TensorArray:
        return await self.session.run(inputs)

    async def load(self) -> bool:
        if self.ready:
            return True

        assert isinstance(self._config, NuxModelConfig)

        # TODO(yan): Wrap functions with async thread now. Replace the functions itself
        self._session = await asynchronous(session.create)(
            self._config.model,
            device=self._config.npu_device,
            compile_config=self._config.compiler_config,
        )
        self._session.run = asynchronous(self._session.run)
        return await super().load()

    async def unload(self):
        if not self.ready:
            return

        self._session.close()
        await super().unload()

    @property
    def session(self) -> session.Session:
        assert self.ready is True, "Could not access session unless model loaded first"
        return self._session

    # TODO(yan): Extract codecs to support other type conversion
    def encode(self, name: str, payload: np.ndarray) -> ResponseOutput:
        return ResponseOutput(
            name=name,
            # TODO(yan): Add datatype dictionary for "BYTES: byte"
            datatype=str(payload.dtype).upper(),
            shape=list(payload.shape),
            data=payload.flatten().tolist(),
        )

    def decode(self, tensor: TensorDesc, request_input: RequestInput) -> np.ndarray:
        return np.array(request_input.data, dtype=tensor.numpy_dtype).reshape(tensor.shape)


class CPUModel(Model):
    """Model runing on CPU."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        predict: Callable[[Any, Any], Awaitable[Any]],
    ):
        super().__init__(config)

        self._predict = predict

    async def predict(self, *args: Any, **kwargs: Any) -> Any:
        return await self._predict(*args, **kwargs)


class OpenVINOModel(Model):
    """Model runing on OpenVINO runtime."""

    from openvino.runtime.ie_api import InferRequest, CompiledModel

    def __init__(self, config: OpenVINOModelConfig, *, compiler_config: Optional[Dict]):

        from openvino.runtime import Core

        super().__init__(config)

        self._runtime = Core()

    async def load(self) -> bool:
        if self.ready:
            return True

        assert isinstance(self._config, OpenVINOModelConfig)

        self._model = self._runtime.compile_model(
            self._runtime.read_model(self._config.model), self._config.compiler_config
        )
        self._request = self._model.create_infer_request()

        # TODO(yan): Wrap functions with async thread now. Use start_async().
        self._request.infer = asynchronous(self._request.infer)

        return await super().load()

    async def predict(self, payload):
        if isinstance(payload, InferenceRequest):
            raise NotImplementedError("OpenVINO model does not support InferenceRequest input.")
        else:
            return await self.session.infer(payload)

    @property
    def session(self) -> InferRequest:
        assert self.ready is True, "Could not access session unless model loaded first"
        return self._request

    @property
    def inner(self) -> CompiledModel:
        return self._model
