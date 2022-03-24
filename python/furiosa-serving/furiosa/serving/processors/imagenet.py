from typing import Dict

# Not yet typed. See https://github.com/python-pillow/Pillow/issues/2625
from PIL import Image  # type: ignore
from fastapi import File, UploadFile
import numpy as np

from .. import ServeModel
from .base import Processor


class ImageNet(Processor):
    def __init__(self, model: ServeModel, label: str):
        self.model = model
        self.label = label

    async def preprocess(self, image: UploadFile = File(...)) -> np.ndarray:  # type: ignore
        """
        Preprocess to convert a image (Python file-like object) to Numpy array
        """

        # Get model input tensor to find out tesnor shape
        _, height, width, channel = self.model.inputs[0].shape

        # Convert PIL image to Numpy array
        data = np.zeros((width, height, channel), np.uint8)
        data[:width, :height, :channel] = (
            Image.open(image.file).convert("RGB").resize((width, height))
        )
        return np.reshape(data, (1, width, height, channel))

    async def postprocess(self, output: np.ndarray) -> Dict:  # type: ignore
        """
        Postprocess to classify image from compiled model with labels
        """

        classified = np.squeeze(output)

        # Load pre-defined labels
        labels = {index: line.strip() for index, line in enumerate(open(self.label).readlines())}

        # Find objects with index which mostly fits
        objects = sorted((score, index[0]) for index, score in np.ndenumerate(classified))[::-1]

        # Return fifth best fit image labels with scores. Note that we are casting int here to
        # convert numpy uin8 type into Python native int type to allow FastAPI serialize result JSON
        return {labels[index]: int(score) for score, index in objects[:5]}
