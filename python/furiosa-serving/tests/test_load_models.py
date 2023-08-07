import pytest

from furiosa.serving import ServeAPI, ServeModel


@pytest.mark.asyncio
async def test_local_file():
    serve = ServeAPI()

    # Load model from local disk
    imagenet = await serve.model("furiosart")(
        "imagenet", location="./examples/assets/models/image_classification.onnx"
    )

    assert isinstance(imagenet, ServeModel)


# TODO: remove this test after support for nux is removed
@pytest.mark.asyncio
async def test_deprecated_nux():
    serve = ServeAPI()

    with pytest.deprecated_call():
        imagenet = await serve.model("nux")(
            "imagenet", location="./examples/assets/models/image_classification.onnx"
        )

        assert isinstance(imagenet, ServeModel)


@pytest.mark.asyncio
async def test_http():
    serve = ServeAPI()

    # Load model from HTTP
    resnet = await serve.model("furiosart")(
        "imagenet",
        location="https://raw.githubusercontent.com/onnx/models/main/vision/classification/resnet/model/resnet50-v1-12.onnx",  # noqa: E501
    )

    assert isinstance(resnet, ServeModel)
