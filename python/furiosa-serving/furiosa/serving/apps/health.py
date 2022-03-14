from fastapi import FastAPI
from fastapi.routing import APIRoute

from furiosa.server import ModelServerError
from furiosa.server.api.rest.app import exception_handler

from .model import endpoints

app = FastAPI(
    routes=[
        # Liveness and readiness
        APIRoute("/live", endpoints.live),
        APIRoute("/ready", endpoints.ready),
        # Server metadata
        APIRoute(
            "/",
            endpoints.metadata,
        ),
    ],
    exception_handlers={ModelServerError: exception_handler},  # type: ignore
)
