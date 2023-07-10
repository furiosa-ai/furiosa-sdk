import pytest

from furiosa.common import transport


@pytest.mark.asyncio
async def test_read(model_file, model_binary):
    with transport.supported("file://") as client:
        assert await client.read(model_file) == model_binary
