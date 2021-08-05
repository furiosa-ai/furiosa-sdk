from fastapi import FastAPI
from fastapi.routing import APIRoute

from furiosa.server import ModelServerError
from furiosa.server.api.rest.app import exception_handler
from furiosa.server.api.rest.endpoints import ModelEndpoints
from furiosa.server.handlers import PredictHandler
from furiosa.server.settings import ServerConfig

from .repository import repository

endpoints = ModelEndpoints(
    PredictHandler(
        # TODO(yan): Accept user configs
        ServerConfig(),
        repository,
    )
)

app = FastAPI(
    routes=[
        APIRoute(
            "/{model_name}/ready",
            endpoints.model_ready,
        ),
        APIRoute(
            "/{model_name}/versions/{model_version}/ready",
            endpoints.model_ready,
        ),
        # Model metadata
        APIRoute(
            "/{model_name}",
            endpoints.model_metadata,
        ),
        APIRoute(
            "/{model_name}/versions/{model_version}",
            endpoints.model_metadata,
        ),
    ],
    exception_handlers={ModelServerError: exception_handler},  # type: ignore
)
