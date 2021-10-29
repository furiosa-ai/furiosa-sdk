from aiohttp import web
import pytest

from furiosa.registry.client.transport import HTTPTransport


@pytest.fixture(scope="function")
async def server(aiohttp_server):
    """
    Running aiohttp application server fixture to serve static files.
    """
    app = web.Application()
    app.router.add_routes([web.static("/", ".")])

    # Run AioHttp test server.
    yield await aiohttp_server(app)


@pytest.fixture(scope="function")
def transport() -> HTTPTransport:
    return HTTPTransport()


@pytest.mark.asyncio
async def test_fetch(transport, artifact_file, artifacts, server):
    # Load from yaml format artifact config.
    assert (
        await transport.fetch(f"http://{server.host}:{server.port}/{artifact_file}")
    ) == artifacts


@pytest.mark.asyncio
async def test_download(transport, model_file, MNISTnet, server):
    assert await transport.download(f"http://{server.host}:{server.port}/{model_file}") == MNISTnet
