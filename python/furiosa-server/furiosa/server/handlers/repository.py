import asyncio

from ..errors import ModelNotFound
from ..model import NuxModel
from ..repository import Repository
from ..settings import ModelConfig
from ..types import (
    RepositoryIndexRequest,
    RepositoryIndexResponse,
    RepositoryIndexResponseItem,
    State,
)


class RepositoryHandler:
    def __init__(self, repository: Repository):
        self._repository = repository

    async def index(self, payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
        model_configs = await self._repository.list()
        items = await asyncio.gather(*(self._to_item(config) for config in model_configs))

        def returnable(item):
            if payload.ready is None:
                return True

            if not payload.ready:
                return item.state != State.READY

            return item.state == State.READY

        return RepositoryIndexResponse(__root__=list(filter(returnable, items)))

    async def _to_item(self, config: ModelConfig) -> RepositoryIndexResponseItem:
        item = RepositoryIndexResponseItem(
            name=config.name,
            state=State.UNKNOWN,
            reason="",
        )

        item.state = await self._get_state(config)
        item.version = config.version
        return item

    async def _get_state(self, config: ModelConfig) -> State:
        try:
            model = await self._repository.get_model(config.name, config.version)
            if model.ready:
                return State.READY
        except ModelNotFound:
            return State.UNAVAILABLE

        # TODO(yan): Support LOADING/UNLOADING states
        return State.UNKNOWN

    async def load(self, name: str) -> bool:
        model_config = await self._repository.find(name)

        # TODO(yan): Support other model implementation
        return await self._repository.load(NuxModel(model_config))

    async def unload(self, name: str) -> bool:
        return await self._repository.unload(name)
