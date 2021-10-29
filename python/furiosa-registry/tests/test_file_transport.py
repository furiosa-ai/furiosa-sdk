import pytest

from furiosa.registry.client import fetch
from furiosa.registry.client.transport import supported


@pytest.mark.asyncio
async def test_fetch(artifacts):
    assert await fetch("file://tests/fixtures") == artifacts


@pytest.mark.asyncio
async def test_read(model_file, model_binary):
    with supported("file://") as transport:
        assert await transport.read(model_file) == model_binary
