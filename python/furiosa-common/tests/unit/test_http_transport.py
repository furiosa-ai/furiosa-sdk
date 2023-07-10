from aiohttp import web
import pytest

from furiosa.common import transport


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
async def test_read(server, model_file, model_binary):
    with transport.supported("http://") as client:
        assert await client.read(f"http://{server.host}:{server.port}/{model_file}") == model_binary
