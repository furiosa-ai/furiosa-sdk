import pytest

import furiosa.registry as registry
from furiosa.registry.client.transport import supported


@pytest.mark.asyncio
async def test_list(artifacts):
    assert await registry.list("file://tests/fixtures") == artifacts


@pytest.mark.asyncio
async def test_read(model_file, model_binary):
    with supported("file://") as transport:
        assert await transport.read(model_file) == model_binary
