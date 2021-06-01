#!/usr/bin/env python3

"""Image segmentation example"""
import sys
import time
import numpy
from pathlib import Path
from PIL import Image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def segment(image_path):
    from furiosa.runtime import session

    image = Image.open(image_path)
    print(type(image))
    # image.show()

    current_path = Path(__file__).absolute().parent
    model_path = current_path.joinpath("../assets/quantized_models/deeplabv3_resnet50_argmax_int8.onnx")

    print(f"Loading and compiling the model {model_path}")
    with session.create(str(model_path)) as sess:
        print(f"Model has been compiled successfully")

        print("Model input and output:")
        print(sess.print_summary())

        _, height, width, channel = sess.input(0).shape()
        image = image.convert('RGB').resize((width, height))

        data = numpy.zeros((width, height, channel), numpy.uint8)
        data[:width, :height, :channel] = image
        data = numpy.reshape(data, (1, width, height, channel))

        start_time = time.time()
        outputs = sess.run(data)
        print(outputs)
        print('Prediction elapsed {:.2f} secs'.format(time.time() - start_time))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("./image_segmentation.py <image>\n")
        sys.exit(-1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        sys.stderr.write(f"{image_path} not found")
        sys.exit(-1)

    segment(image_path)