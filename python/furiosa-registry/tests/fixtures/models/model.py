"""Code format model fixture."""
from furiosa.registry import Model
from furiosa.registry.client.transport import FileTransport

loader = FileTransport()


class MLCommonsModel(Model):
    def preprocess(self):
        pass

    def postprocess(self):
        pass


async def mlcommons_ssd_resnet34_int8(message):
    model = MLCommonsModel(
        name="mlcommons_ssd_resnet34_int8",
        # We just use dummy model data.
        model=await loader.read("models/MNISTnet_uint8_quant_without_softmax.tflite"),
    )

    return model
