import textwrap
from typing import Dict

from fastapi import UploadFile
import numpy as np
import pytest

from furiosa.serving import ServeAPI, ServeModel
from furiosa.serving.processors import ImageNet


@pytest.mark.asyncio
async def test_imagenet():
    serve = ServeAPI()

    model: ServeModel = await serve.model(
        "imagenet", location="./examples/assets/models/image_classification.onnx"
    )

    @model.post("/models/imagenet/infer")
    @ImageNet(
        model=model, label="./examples/assets/labels/ImageNetLabels.txt"
    )  # This makes infer() Callable[[UploadFile], Dict]
    async def infer(tensor: np.ndarray) -> np.ndarray:
        """Actual inference"""
        return await model.predict(tensor)

    doc = """
    Preprocess to convert a image (Python file-like object) to Numpy array
    Actual inference
    Postprocess to classify image from compiled model with labels
    """

    assert infer.__annotations__["image"] == UploadFile
    assert infer.__annotations__["return"] == Dict
    assert infer.__doc__.split() == doc.split()
