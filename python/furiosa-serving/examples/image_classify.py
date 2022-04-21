from fastapi import FastAPI
import numpy as np

from furiosa.common.thread import synchronous
from furiosa.serving import ServeAPI, ServeModel
from furiosa.serving.apps import health, model, repository
from furiosa.serving.processors import ImageNet

# Main serve API
serve = ServeAPI(repository.repository)

# This is FastAPI instance
app: FastAPI = serve.app

# Define model
network: ServeModel = synchronous(serve.model)(
    "imagenet", location="./assets/models/image_classification.onnx"
)


@network.post("/imagenet/infer")
@ImageNet(model=network, label="./assets/labels/ImageNetLabels.txt")
async def infer(tensor: np.ndarray) -> np.ndarray:
    """
    Infer from model runtime with speicified tensor
    """
    tensors = [tensor for tensor in await network.predict([tensor])]
    return tensors[0]


app.mount("/repository", repository.app)
app.mount("/models", model.app)
app.mount("/health", health.app)
