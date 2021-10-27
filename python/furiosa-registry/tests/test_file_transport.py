import pytest

from furiosa.registry.client.transport import FileTransport


@pytest.fixture(scope="function")
def transport() -> FileTransport:
    return FileTransport()


@pytest.mark.asyncio
async def test_fetch(transport, artifact_file, artifacts):
    assert await transport.fetch(artifact_file) == artifacts


@pytest.mark.asyncio
async def test_download(transport, model_file, MNISTnet):
    assert await transport.download(model_file) == MNISTnet
