import asyncio
import numpy as np

from furiosa.server.model import NPUModel, NPUModelConfig

class SimpleApplication:
    def __init__(self):
        self.model = NPUModel(
            NPUModelConfig(
                name="MNIST",
                model="mnist.onnx",
            )
        )

    async def load(self):
        await self.model.load()

    async def process(self, image):
        input = self.preprocess(image)
        tensor = await self.model.predict(input)
        output = self.postprocess(tensor)
        return output

    def preprocess(self, image):
        # do preprocess
        return image

    def postprocess(self, tensor):
        # do postprocess
        return tensor


APP = SimpleApplication()

async def startup():
    await APP.load()

async def run(image):
    result = await APP.process(image)
    return result

if __name__ == "__main__":
    asyncio.run(startup())

    image = np.random.rand(1, 1, 28, 28).astype(np.float32)
    asyncio.run(run(image))