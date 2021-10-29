from aiohttp import web
import pytest

from furiosa.registry.client import fetch
from furiosa.registry.client.transport import supported


@pytest.fixture(scope="function")
async def server(aiohttp_server):
    """
    Running aiohttp application server fixture to serve static files.
    """
    app = web.Application()
    app.router.add_routes([web.static("/", ".")])

    # Run AioHttp test server.
    yield await aiohttp_server(app)


@pytest.mark.asyncio
async def test_fetch(server, tflite_artifact, artifacts):
    # Load from yaml format artifact config.
    assert (await fetch(f"http://{server.host}:{server.port}/tests/fixtures")) == artifacts


@pytest.mark.asyncio
async def test_read(server, model_file, model_binary):
    with supported("http://") as transport:
        assert (
            await transport.read(f"http://{server.host}:{server.port}/{model_file}") == model_binary
        )
