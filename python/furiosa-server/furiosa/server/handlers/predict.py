from ..repository import Repository
from ..settings import ServerConfig
from ..types import (
    InferenceRequest,
    InferenceResponse,
    MetadataModelResponse,
    MetadataServerResponse,
)


class PredictHandler:
    def __init__(self, config: ServerConfig, repository: Repository):
        self._config = config
        self._repository = repository

    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        models = await self._repository.get_models()
        return all(model.ready for model in models)

    async def model_ready(self, name: str, version: str = None) -> bool:
        model = await self._repository.get_model(name, version)
        return model.ready

    async def metadata(self) -> MetadataServerResponse:
        return MetadataServerResponse(
            name=self._config.server_name,
            version=self._config.server_version,
            extensions=self._config.extensions,
        )

    async def model_metadata(self, name: str, version: str = None) -> MetadataModelResponse:
        model = await self._repository.get_model(name, version)
        return await model.metadata()

    async def infer(
        self, payload: InferenceRequest, name: str, version: str = None
    ) -> InferenceResponse:
        model = await self._repository.get_model(name, version)

        prediction = await model.predict(payload)
        prediction.id = payload.id

        return prediction
