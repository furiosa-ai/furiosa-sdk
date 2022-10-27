"""Model class for prediction/explanation."""
from abc import ABC, abstractmethod
import asyncio
from asyncio import Future
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    overload,
)
import uuid

import numpy as np

from furiosa.common.thread import asynchronous
from furiosa.runtime import envs, session
from furiosa.runtime.errors import QueueWaitTimeout
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

        # Uninitialized session pool. Will be loaded in load().
        self.pool: Optional[Iterator[Union[session.Session, session.AsyncSession]]] = None
        self.sessions: Optional[List[Union[session.Session, session.AsyncSession]]] = None

    async def predict(self, payload):
        if isinstance(payload, InferenceRequest):
            # Python list to Numpy array
            inputs = [
                self.decode(tensor, request)
                for tensor, request in zip(self.session.inputs(), payload.inputs)
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
        assert self.pool is not None

        session = next(self.pool)

        assert session is not None

        return await session.run(inputs)  # type: ignore

    async def load(self) -> bool:
        if self.ready:
            return True

        assert isinstance(self._config, NuxModelConfig)

        devices = self._config.npu_device or envs.current_npu_device()

        self.sessions = await self.create_sessions(devices)

        # Round robin naive session pool
        self.pool = itertools.cycle(self.sessions)

        return await super().load()

    async def create_sessions(
        self, devices: str
    ) -> List[Union[session.Session, session.AsyncSession]]:
        create = asynchronous(session.create)

        sessions = []
        for device in devices.split(","):
            blocking = await create(
                self._config.model,  # type: ignore
                device=device,
                batch_size=self._config.batch_size,  # type: ignore
                worker_num=self._config.worker_num,  # type: ignore
                compiler_config=self._config.compiler_config,  # type: ignore
            )
            blocking.run = asynchronous(blocking.run)
            sessions.append(blocking)

        return sessions

    async def unload(self):
        if not self.ready:
            return

        for s in self.sessions:
            s.close()

        await super().unload()

    @property
    def session(self) -> session.Session:
        assert self.ready is True, "Could not access session unless model loaded first"
        # First session
        return next(iter(self.sessions))  # type: ignore

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


class AsyncNuxModel(NuxModel):
    """Model Nux runtime based on AsyncSession."""

    def __init__(self, config: NuxModelConfig):
        super().__init__(config)

        self.queue: Dict[uuid.UUID, object] = {}

    def completed(self, receiver: session.CompletionQueue, id: uuid.UUID, future: Future):
        try:
            context, value = receiver.recv(1)
        except QueueWaitTimeout:
            context, value = None, None
            pass

        # Save response
        if context is not None:
            self.queue[context] = value

        if id in self.queue:
            # Response found
            future.set_result(self.queue.pop(id))
            return

        # Try next time
        loop = asyncio.get_event_loop()
        loop.call_soon(self.completed, receiver, id, future)

    async def run(self, inputs: Sequence[np.ndarray]) -> TensorArray:
        sender, receiver = next(self.pool)  # type: ignore

        id = uuid.uuid1()

        sender.submit(inputs, id)

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        loop.call_soon(self.completed, receiver, id, future)

        return await future

    async def create_sessions(
        self, devices: str
    ) -> List[Union[session.Session, session.AsyncSession]]:
        create = asynchronous(session.create_async)

        sessions = []
        for device in devices.split(","):
            unblocking = await create(
                self._config.model,
                device=device,
                batch_size=self._config.batch_size,
                worker_num=self._config.worker_num,
                compiler_config=self._config.compiler_config,
            )
            sessions.append(unblocking)

        return sessions


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

    if TYPE_CHECKING:
        from openvino.runtime.ie_api import CompiledModel, InferRequest

    def __init__(self, config: OpenVINOModelConfig):
        from openvino.runtime import Core

        super().__init__(config)

        self._runtime = Core()

    async def load(self) -> bool:
        if self.ready:
            return True

        assert isinstance(self._config, OpenVINOModelConfig)

        self._model = self._runtime.compile_model(
            self._runtime.read_model(self._config.model), "CPU", self._config.compiler_config
        )
        self._request = self._model.create_infer_request()
        return await super().load()

    async def predict(self, payload):
        """Inference via OpenVINO runtime.

        Note that it's not thread safe API as OpenVINO API does not support.
        """
        if isinstance(payload, InferenceRequest):
            raise NotImplementedError("OpenVINO model does not support InferenceRequest input.")
        else:
            self.session.start_async(payload)

            while True:
                # Check whether the previous request is done.
                # Use wait_for() as InferRequest does not have explict API to do.
                if self.session.wait_for(1):
                    return self.session.results

                # 1 us
                await asyncio.sleep(0.000001)

    @property
    def session(self) -> "InferRequest":
        assert self.ready is True, "Could not access session unless model loaded first"
        return self._request

    @property
    def inner(self) -> "CompiledModel":
        return self._model
