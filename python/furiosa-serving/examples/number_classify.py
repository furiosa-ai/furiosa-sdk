import asyncio
import io
import logging
import os
import re
from typing import Dict, List

from PIL import Image
from fastapi import UploadFile
import numpy as np
from opentelemetry import trace
import uvicorn

from furiosa.common.thread import synchronous
from furiosa.serving import NPUServeModel, ServeAPI

# Main serve API
serve = ServeAPI()

# This is FastAPI instance
app = serve.app
tracer = trace.get_tracer(__name__)


def detect_npu_devices(use_fusion: bool) -> List[str]:
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
    npu_device=",".join(
        detect_npu_devices(True)
    ),  # or specify device name: e.g., "npu0pe0,npu1pe0"
    worker_num=4,
)


class Application:
    def __init__(self, model: NPUServeModel):
        self.model = model

    async def process(self, image: Image.Image) -> int:
        with tracer.start_as_current_span("preprocess"):
            input_tensors = self.preprocess(image)
        with tracer.start_as_current_span("inference"):
            output_tensors = await self.inference(input_tensors)
        with tracer.start_as_current_span("postprocess"):
            return self.postprocess(output_tensors)

    @staticmethod
    def preprocess(image: Image.Image) -> List[np.ndarray]:
        logging.info("in preprocessing")
        origin_tensor = np.array(image)
        number_count = int(origin_tensor.shape[1] / 28)

        return [
            tensor.reshape(1, 28, 28, 1) for tensor in np.split(origin_tensor, number_count, axis=1)
        ]

    async def inference(self, tensors: List[np.ndarray]) -> List[np.ndarray]:
        logging.info("in inferencing")
        return await asyncio.gather(*(self.model.predict(tensor) for tensor in tensors))

    @staticmethod
    def postprocess(tensors: List[np.ndarray]) -> str:
        logging.info("in postprocessing")
        res = [str(np.argmax(tensor[0].reshape(1, 10))) for tensor in tensors]
        return "".join(res)


application: Application = Application(model)


@app.post("/infer")
async def infer(image: UploadFile) -> Dict:
    logging.info("Start inference process.")
    with tracer.start_as_current_span("Image.open"):
        image: Image.Image = Image.open(io.BytesIO(await image.read()))
    with tracer.start_as_current_span("application:process"):
        result = await application.process(image)
        logging.info(f"Result value: {result}")
        return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
