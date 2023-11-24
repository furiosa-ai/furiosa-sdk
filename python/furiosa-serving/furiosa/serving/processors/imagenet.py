from typing import Dict, Tuple

# Not yet typed. See https://github.com/python-pillow/Pillow/issues/2625
from PIL import Image  # type: ignore
from fastapi import File, UploadFile
import numpy as np
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


async def preprocess(shape: Tuple[int, ...], image: UploadFile = File(...)) -> np.ndarray:
    """
    Preprocess to convert a image (Python file-like object) to Numpy array
    """

    # Get model input tensor to find out tesnor shape
    _, height, width, channel = shape

    # Convert PIL image to Numpy array
    with tracer.start_as_current_span("zeros"):
        data = np.zeros((width, height, channel), np.uint8)
    with tracer.start_as_current_span("convert and resize"):
        data[:width, :height, :channel] = (
            Image.open(image.file).convert("RGB").resize((width, height))
        )
    with tracer.start_as_current_span("reshape"):
        result = np.reshape(data, (1, width, height, channel))
    return result


async def postprocess(output: np.ndarray, label: str) -> Dict:
    """
    Postprocess to classify image from compiled model with labels
    """

    with tracer.start_as_current_span("squeeze"):
        classified = np.squeeze(output)

    # Load pre-defined labels
    with tracer.start_as_current_span("load labels"):
        labels = {index: line.strip() for index, line in enumerate(open(label).readlines())}

    # Find objects with index which mostly fits
    with tracer.start_as_current_span("find objects"):
        objects = sorted((score, index[0]) for index, score in np.ndenumerate(classified))[::-1]

    # Return fifth best fit image labels with scores. Note that we are casting int here to
    # convert numpy uin8 type into Python native int type to allow FastAPI serialize result JSON
    with tracer.start_as_current_span("write result"):
        result = {labels[index]: int(score) for score, index in objects[:5]}
    return result
