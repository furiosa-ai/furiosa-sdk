import pytest

from furiosa.registry.client import listing
from furiosa.registry.client.transport import supported


@pytest.mark.asyncio
async def test_listing(artifacts):
    assert await listing("file://tests/fixtures") == artifacts


@pytest.mark.asyncio
async def test_read(model_file, model_binary):
    with supported("file://") as transport:
        assert await transport.read(model_file) == model_binary
