import asyncio
import io
from typing import Dict, List

from PIL import Image
from fastapi import UploadFile
import numpy as np
import uvicorn

from furiosa.common.thread import synchronous
from furiosa.serving import NPUServeModel, ServeAPI

# Main serve API
serve = ServeAPI()

# This is FastAPI instance
app = serve.app

# Define model
model: NPUServeModel = synchronous(serve.model("nux"))(
    "MNIST",
    location="./assets/models/MNISTnet_uint8_quant_without_softmax.tflite",
    worker_num=4,
)


class Application:
    def __init__(self, model: NPUServeModel):
        self.model = model

    async def process(self, image: Image.Image) -> int:
        input_tensors = self.preprocess(image)
        output_tensors = await self.inference(input_tensors)
        return self.postprocess(output_tensors)

    @staticmethod
    def preprocess(image: Image.Image) -> List[np.ndarray]:
        origin_tensor = np.array(image)
        number_count = int(origin_tensor.shape[1] / 28)

        return [
            tensor.reshape(1, 28, 28, 1) for tensor in np.split(origin_tensor, number_count, axis=1)
        ]

    async def inference(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        return await asyncio.gather(*(self.model.predict(tensor) for tensor in tensors))

    @staticmethod
    def postprocess(tensors: List[np.ndarray]) -> str:
        res = [str(np.argmax(tensor[0].reshape(1, 10))) for tensor in tensors]
        return "".join(res)


application: Application = Application(model)


@app.post("/infer")
async def infer(image: UploadFile) -> Dict:
    image: Image.Image = Image.open(io.BytesIO(await image.read()))
    return {"result": await application.process(image)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
