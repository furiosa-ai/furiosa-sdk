import uvicorn

from ...handlers import PredictHandler, RepositoryHandler
from ...settings import RESTServerConfig
from .app import create


class RESTServer:
    def __init__(
        self,
        config: RESTServerConfig,
        predict_handler: PredictHandler,
        repository_handler: RepositoryHandler,
    ):
        self._config = config
        self._app = create(
            self._config,
            predict_handler=predict_handler,
            repository_handler=repository_handler,
        )

    async def start(self):
        class NoSignalServer(uvicorn.Server):
            def install_signal_handlers(self):
                # Delegate signal handling to ModelServer
                pass

        self._server = NoSignalServer(
            uvicorn.Config(
                self._app,
                host=self._config.host,
                port=self._config.port,
                workers=self._config.workers,
            )
        )
        await self._server.serve()

    async def stop(self):
        self._server.handle_exit(sig=None, frame=None)
