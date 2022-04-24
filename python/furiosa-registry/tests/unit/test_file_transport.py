import pytest

import furiosa.registry as registry
from furiosa.registry.client.transport import supported


@pytest.mark.asyncio
async def test_list():
    assert await registry.list("file://tests/unit/fixtures") == ["MNISTNet"]


@pytest.mark.asyncio
async def test_load():
    assert (await registry.load("file://tests/unit/fixtures", "MNISTNet")).name == "MNISTNet"


@pytest.mark.asyncio
async def test_help():
    assert (await registry.help("file://tests/unit/fixtures", "MNISTNet")) == "MNISTNet Model"


@pytest.mark.asyncio
async def test_read(model_file, model_binary):
    with supported("file://") as transport:
        assert await transport.read(model_file) == model_binary
