from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile

from furiosa.common.thread import synchronous
from furiosa.serving import ServeAPI, ServeModel
from furiosa.serving.apps import health, model, repository
from furiosa.serving.processors.imagenet import postprocess, preprocess

current_directory = Path(__file__).parent


# Main serve API
serve = ServeAPI(repository.repository)

# This is FastAPI instance
app: FastAPI = serve.app

# Define model
network: ServeModel = synchronous(serve.model("furiosart"))(
    "imagenet", location=str(current_directory / "assets/models/image_classification.onnx")
)


@network.post("/imagenet/infer")
async def infer(image: UploadFile = File(...)) -> Dict:
    """
    Infer from model runtime with speicified tensor
    """
    shape = network.inputs[0].shape
    tensor = await preprocess(shape, image)
    tensors = [tensor for tensor in await network.predict([tensor])]
    results = await postprocess(
        tensors[0], label=str(current_directory / "assets/labels/ImageNetLabels.txt")
    )
    return results


app.mount("/repository", repository.app)
app.mount("/models", model.app)
app.mount("/health", health.app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
