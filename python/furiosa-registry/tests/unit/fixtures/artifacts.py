from typing import Any

from furiosa.registry import Format, Metadata, Model, Publication
from furiosa.registry.client.transport import FileTransport

loader = FileTransport()


class MNISTNetModel(Model):
    """MNISTNet Model"""

    ...


async def MNISTNet(*args: Any, **kwargs: Any) -> MNISTNetModel:
    return MNISTNetModel(
        name="MNISTNet",
        source=await loader.read("models/MNISTnet_uint8_quant_without_softmax.tflite"),
        dfg=await loader.read("models/MNISTnet_uint8_quant_without_softmax.dfg"),
        enf=await loader.read("models/MNISTnet_uint8_quant_without_softmax.enf"),
        format=Format.TFLite,
        family="MNIST",
        version="v1.0",
        metadata=Metadata(
            description="MNIST quantized model",
        ),
        publication=Publication(url="https://en.wikipedia.org/wiki/MNIST_database"),
        **kwargs,
    )
