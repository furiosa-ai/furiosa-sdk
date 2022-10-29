import asyncio
import io
import os
import re
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


def detect_npu_devices(use_fusion: bool) -> List[str]:
    """This function tries to find all NPU device files from /dev and return the list."""
    # If you want to knows how to specify devices, please refer to the section 'How to Specify a NPU device'
    # at https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/notebooks/AdvancedTopicsInInferenceAPIs.ipynb
    if use_fusion:
        PATTERN = re.compile("npu[0-9]+pe0-1")
    else:
        PATTERN = re.compile("npu[0-9]+pe[0|1]$")

    results = sorted(entry.name for entry in os.scandir("/dev") if PATTERN.match(entry.name))

    if not results:
        raise RuntimeError("No NPU device detected.")

    return results


# Define model
model: NPUServeModel = synchronous(serve.model("nux"))(
    "MNIST",
    location="./assets/models/MNISTnet_uint8_quant_without_softmax.tflite",
    # A Model can have a pool of multiple devices.
    npu_device=",".join(
        detect_npu_devices(True)
    ),  # or specify device name: e.g., "npu0pe0,npu1pe0"
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

        # this result array will be passed to inference()
        return [
            tensor.reshape(1, 28, 28, 1) for tensor in np.split(origin_tensor, number_count, axis=1)
        ]

    async def inference(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        # The following code runs multiple inferences at the same time and wait until all requests are completed.
        return await asyncio.gather(*(self.model.predict(tensor) for tensor in tensors))

    @staticmethod
    def postprocess(tensors: List[np.ndarray]) -> str:
        characters = [str(np.argmax(tensor[0].reshape(1, 10))) for tensor in tensors]
        return "".join(characters)


application: Application = Application(model)


@app.post("/infer")
async def infer(image: UploadFile) -> Dict:
    image: Image.Image = Image.open(io.BytesIO(await image.read()))

    return {"result": await application.process(image)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
