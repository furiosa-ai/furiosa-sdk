import pytest

from furiosa.registry.client.resolver import resolve


@pytest.mark.asyncio
async def test_resolve(tflite_artifact, tflite_model):
    model = await resolve("file://tests/fixtures", tflite_artifact)
    assert model == tflite_model
