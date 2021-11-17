import pytest

from furiosa.registry.client import resolve


@pytest.mark.asyncio
async def test_resolve(code_artifact, code_model):
    model = await resolve("file://tests/fixtures", code_artifact, message="Argument for the Model")

    assert model == code_model

    assert hasattr(model, "preprocess")
    assert hasattr(model, "postprocess")
