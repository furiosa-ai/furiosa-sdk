"""Model class for prediction/explanation."""
from abc import ABC, abstractmethod
import asyncio
from typing import TYPE_CHECKING, Generic, List, Optional, TypeVar, Union, overload

import numpy as np
from typing_extensions import deprecated

from furiosa.runtime import Runner, Tensor, TensorArray, TensorDesc, create_runner
from furiosa.runtime._utils import default_device

from .settings import FuriosaRTModelConfig, ModelConfig, OpenVINOModelConfig
from .types import (
    InferenceRequest,
    InferenceResponse,
    MetadataModelResponse,
    RequestInput,
    ResponseOutput,
)

T = TypeVar("T", bound=ModelConfig)


class Model(ABC, Generic[T]):
    """Base model class for every runtime."""

    def __init__(self, config: T):
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

    @overload
    async def predict(self, payload: np.ndarray) -> List[np.ndarray]:
        ...

    @abstractmethod
    async def predict(self, payload):
        pass


class FuriosaRTModel(Model[FuriosaRTModelConfig]):
    """Model running on NPU."""

    def __init__(self, config: FuriosaRTModelConfig):
        super().__init__(config)

        self.runner: Optional[Runner] = None

    async def predict(self, payload):
        if isinstance(payload, InferenceRequest):
            # Python list to Numpy array
            inputs = [
                self.decode(tensor, request)
                for tensor, request in zip(self.runner.model.inputs(), payload.inputs)
            ]
        else:
            inputs = payload

        tensors = await self.run(inputs)

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

    async def run(self, inputs: Union[Tensor, TensorArray]) -> Union[Tensor, TensorArray]:
        assert self.ready
        assert self.runner is not None

        return await self.runner.run(inputs)

    async def load(self) -> bool:
        if self.ready:
            return True

        assert isinstance(self._config, FuriosaRTModelConfig)

        devices = self._config.npu_device or default_device()
        assert devices is not None

        self.runner = await create_runner(
            self._config.model,
            device=devices,
            batch_size=self._config.batch_size,
            worker_num=self._config.worker_num,
            compiler_config=self._config.compiler_config,
        )

        return await super().load()

    async def unload(self):
        if not self.ready:
            return

        await self.runner.close()
        await super().unload()

    # TODO(yan): Extract codecs to support other type conversion
    def encode(self, name: str, payload: np.ndarray) -> ResponseOutput:
        return ResponseOutput(
            name=name,
            # TODO(yan): Add datatype dictionary for "BYTES: byte"
            datatype=str(payload.dtype).upper(),
            shape=list(payload.shape),
            data=payload.flatten().tolist(),
        )

    def decode(self, tensor_desc: TensorDesc, request_input: RequestInput) -> np.ndarray:
        return np.array(request_input.data, dtype=tensor_desc.dtype.numpy).reshape(
            tensor_desc.shape
        )


@deprecated("Use NPUModel instead")
class NuxModel(FuriosaRTModel):
    ...


@deprecated("Use NPUModel instead")
class AsyncNuxModel(FuriosaRTModel):
    ...


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
