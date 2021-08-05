from fastapi import FastAPI
from fastapi.routing import APIRoute

from furiosa.server import ModelServerError
from furiosa.server.api.rest.app import exception_handler
from furiosa.server.api.rest.endpoints import ModelRepositoryEndpoints
from furiosa.server.handlers import RepositoryHandler
from furiosa.server.registry import InMemoryRegistry
from furiosa.server.repository import Repository

registry = InMemoryRegistry()

repository = Repository([registry])

endpoints = ModelRepositoryEndpoints(RepositoryHandler(repository))

app = FastAPI(
    routes=[
        APIRoute(
            "/index",
            endpoints.index,
            methods=["POST"],
        ),
        APIRoute(
            "/models/{model_name}/load",
            endpoints.load,
            methods=["POST"],
        ),
        APIRoute(
            "/models/{model_name}/unload",
            endpoints.unload,
            methods=["POST"],
        ),
    ],
    exception_handlers={ModelServerError: exception_handler},  # type: ignore
)
