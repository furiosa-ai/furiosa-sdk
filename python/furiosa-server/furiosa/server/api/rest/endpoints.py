from fastapi import status
from fastapi.responses import Response

from ...handlers import PredictHandler, RepositoryHandler
from ...types import (
    InferenceRequest,
    InferenceResponse,
    MetadataModelResponse,
    MetadataServerResponse,
    RepositoryIndexRequest,
    RepositoryIndexResponse,
)


class BooleanResponse(Response):
    def __init__(self, response: bool, error_code: int = status.HTTP_400_BAD_REQUEST):
        status_code = status.HTTP_200_OK if response else error_code
        super().__init__(status_code=status_code)


class ModelEndpoints:
    def __init__(self, handler: PredictHandler):
        self._handler = handler

    async def live(self) -> Response:
        return BooleanResponse(await self._handler.live())

    async def ready(self) -> Response:
        return BooleanResponse(await self._handler.ready())

    async def model_ready(self, model_name: str, model_version: str = None) -> Response:
        return BooleanResponse(await self._handler.model_ready(model_name, model_version))

    async def metadata(self) -> MetadataServerResponse:
        return await self._handler.metadata()

    async def model_metadata(
        self, model_name: str, model_version: str = None
    ) -> MetadataModelResponse:
        return await self._handler.model_metadata(model_name, model_version)

    async def infer(
        self,
        payload: InferenceRequest,
        model_name: str,
        model_version: str = None,
    ) -> InferenceResponse:
        return await self._handler.infer(payload, model_name, model_version)


class ModelRepositoryEndpoints:
    def __init__(self, handler: RepositoryHandler):
        self._handler = handler

    async def index(self, payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
        return await self._handler.index(payload)

    async def load(self, model_name: str) -> Response:
        return BooleanResponse(await self._handler.load(name=model_name))

    async def unload(self, model_name: str) -> Response:
        return BooleanResponse(await self._handler.unload(name=model_name))
