from typing import Callable

import grpc

from ...errors import ModelServerError
from ...handlers import PredictHandler, RepositoryHandler
from .converters import (
    ModelInferRequestConverter,
    ModelInferResponseConverter,
    ModelMetadataResponseConverter,
    RepositoryIndexRequestConverter,
    RepositoryIndexResponseConverter,
    ServerMetadataResponseConverter,
)
from .generated import model_repository_pb2 as mr_pb
from .generated import predict_pb2 as pb
from .generated.model_repository_pb2_grpc import ModelRepositoryServiceServicer
from .generated.predict_pb2_grpc import GRPCInferenceServiceServicer


def _handle_error(f: Callable):
    async def _inner(self, request, context):
        try:
            return await f(self, request, context)
        except ModelServerError as err:
            await context.abort(code=grpc.StatusCode.INVALID_ARGUMENT, details=str(err))

    return _inner


class InferenceServicer(GRPCInferenceServiceServicer):
    def __init__(self, predict_handler: PredictHandler):
        super().__init__()
        self._predict_handler = predict_handler

    async def ServerLive(self, request: pb.ServerLiveRequest, context) -> pb.ServerLiveResponse:
        is_live = await self._predict_handler.live()
        return pb.ServerLiveResponse(live=is_live)

    async def ServerReady(self, request: pb.ServerReadyRequest, context) -> pb.ServerReadyResponse:
        is_ready = await self._predict_handler.ready()
        return pb.ServerReadyResponse(ready=is_ready)

    async def ModelReady(self, request: pb.ModelReadyRequest, context) -> pb.ModelReadyResponse:
        is_model_ready = await self._predict_handler.model_ready(
            name=request.name, version=request.version
        )
        return pb.ModelReadyResponse(ready=is_model_ready)

    async def ServerMetadata(
        self, request: pb.ServerMetadataRequest, context
    ) -> pb.ServerMetadataResponse:
        metadata = await self._predict_handler.metadata()
        return ServerMetadataResponseConverter.from_types(metadata)

    @_handle_error
    async def ModelMetadata(
        self, request: pb.ModelMetadataRequest, context
    ) -> pb.ModelMetadataResponse:
        metadata = await self._predict_handler.model_metadata(
            name=request.name, version=request.version
        )
        return ModelMetadataResponseConverter.from_types(metadata)

    @_handle_error
    async def ModelInfer(self, request: pb.ModelInferRequest, context) -> pb.ModelInferResponse:
        payload = ModelInferRequestConverter.to_types(request)
        result = await self._predict_handler.infer(
            payload=payload, name=request.model_name, version=request.model_version
        )
        response = ModelInferResponseConverter.from_types(result)
        return response


class ModelRepositoryServicer(ModelRepositoryServiceServicer):
    def __init__(self, handler: RepositoryHandler):
        self._handler = handler

    async def RepositoryIndex(
        self, request: mr_pb.RepositoryIndexRequest, context
    ) -> mr_pb.RepositoryIndexResponse:
        payload = RepositoryIndexRequestConverter.to_types(request)
        index = await self._handler.index(payload)
        return RepositoryIndexResponseConverter.from_types(index)

    @_handle_error
    async def RepositoryModelLoad(
        self, request: mr_pb.RepositoryModelLoadRequest, context
    ) -> mr_pb.RepositoryModelLoadResponse:
        await self._handler.load(request.model_name)
        return mr_pb.RepositoryModelLoadResponse()

    @_handle_error
    async def RepositoryModelUnload(
        self, request: mr_pb.RepositoryModelUnloadRequest, context
    ) -> mr_pb.RepositoryModelUnloadResponse:
        await self._handler.unload(request.model_name)
        return mr_pb.RepositoryModelUnloadResponse()
