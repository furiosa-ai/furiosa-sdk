import asyncio
import signal
from typing import List

from .api.grpc.server import GRPCServer
from .api.rest.server import RESTServer
from .handlers import PredictHandler, RepositoryHandler
from .model import NuxModel
from .registry import FileRegistry, InMemoryRegistry
from .repository import Repository
from .settings import ModelConfig, ServerConfig


class ModelServer:
    def __init__(self, config: ServerConfig, model_configs: List[ModelConfig]):
        self._config = config
        self._model_configs = model_configs

        self._repository = Repository(
            [FileRegistry(config.file_registry_config), InMemoryRegistry(model_configs)]
        )

        self._predict_handler = PredictHandler(config, self._repository)
        self._repository_handler = RepositoryHandler(self._repository)

        self._rest_server = RESTServer(
            config.rest_server_config, self._predict_handler, self._repository_handler
        )

        self._grpc_server = GRPCServer(
            config.grpc_server_config, self._predict_handler, self._repository_handler
        )

    async def start(self):
        # Signal handling
        loop = asyncio.get_event_loop()
        for sign in (signal.SIGTERM, signal.SIGINT):
            # Add signal handler for terminating GRPC/REST server at once
            loop.add_signal_handler(sign, lambda: asyncio.ensure_future(self.stop()))

        # Model loading
        if self._config.intial_model_autoload:
            await self.load()

        # Run servers
        await asyncio.gather(self._rest_server.start(), self._grpc_server.start())

    async def stop(self):
        await asyncio.gather(self._rest_server.stop(), self._grpc_server.stop())

    async def load(self):
        for config in await self._repository.list():
            # TODO(yan): Support other model implementation
            await self._repository.load(NuxModel(config))
