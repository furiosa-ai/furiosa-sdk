from fastapi import FastAPI
import pytest
from starlette.routing import Mount

from furiosa.serving import ServeAPI
from furiosa.serving.apps import health, model, repository


@pytest.mark.asyncio
async def test_app_mount():
    # Create ServeAPI with Repository instance. This repository maintains models
    serve = ServeAPI(repository.repository)

    app: FastAPI = serve.app

    app.mount("/repository", repository.app)
    app.mount("/models", model.app)
    app.mount("/health", health.app)

    assert len([route for route in app.routes if isinstance(route, Mount)]) == 3
