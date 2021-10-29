from concurrent.futures import ThreadPoolExecutor

from grpc import aio

from ...handlers import PredictHandler, RepositoryHandler
from ...settings import GRPCServerConfig
from .generated.model_repository_pb2_grpc import add_ModelRepositoryServiceServicer_to_server
from .generated.predict_pb2_grpc import add_GRPCInferenceServiceServicer_to_server
from .servicers import InferenceServicer, ModelRepositoryServicer


class GRPCServer:
    def __init__(
        self,
        config: GRPCServerConfig,
        predict_handler: PredictHandler,
        repository_handler: RepositoryHandler,
    ):
        self._config = config
        self._predict_handler = predict_handler
        self._repository_handlers = repository_handler

    def _create_server(self):
        self._inference_servicer = InferenceServicer(self._predict_handler)
        self._model_repository_servicer = ModelRepositoryServicer(self._repository_handlers)
        self._server = aio.server(ThreadPoolExecutor(max_workers=self._config.workers))

        add_GRPCInferenceServiceServicer_to_server(self._inference_servicer, self._server)
        add_ModelRepositoryServiceServicer_to_server(self._model_repository_servicer, self._server)

        self._server.add_insecure_port(f"{self._config.host}:{self._config.port}")

        return self._server

    async def start(self):
        self._create_server()

        await self._server.start()
        await self._server.wait_for_termination()

    async def stop(self):
        await self._server.stop(grace=5)
