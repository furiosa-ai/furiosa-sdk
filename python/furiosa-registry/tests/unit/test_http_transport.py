from aiohttp import web
import pytest

import furiosa.registry as registry
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
async def test_list(server):
    # HTTP transport download() not yet implemented
    with pytest.raises(NotImplementedError):
        assert (await registry.list(f"http://{server.host}:{server.port}/tests/unit/fixtures")) == [
            "MNISTNet"
        ]


@pytest.mark.asyncio
async def test_load(server):
    # HTTP transport download() not yet implemented
    with pytest.raises(NotImplementedError):
        assert (
            await registry.load(
                f"http://{server.host}:{server.port}/tests/unit/fixtures", "MNISTNet"
            )
        ).name == "MNISTNet"


@pytest.mark.asyncio
async def test_help(server):
    # HTTP transport download() not yet implemented
    with pytest.raises(NotImplementedError):
        assert (
            await registry.help(
                f"http://{server.host}:{server.port}/tests/unit/fixtures", "MNISTNet"
            )
        ) == "MNISTNet Model"


@pytest.mark.asyncio
async def test_read(server, model_file, model_binary):
    with supported("http://") as transport:
        assert (
            await transport.read(f"http://{server.host}:{server.port}/{model_file}") == model_binary
        )
