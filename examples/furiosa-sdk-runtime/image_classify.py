#!/usr/bin/env python3

"""Image classification example"""
import collections
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


def classify(image_path):
    from furiosa.runtime import session
    from helper import load_labels

    image = Image.open(image_path)
    current_path = Path(__file__).absolute().parent
    path = current_path \
        .joinpath("../assets/quantized_models/imagenet_224x224_mobilenet_v1_uint8_quantization-aware-trained_dm_1.0_without_softmax.tflite")

    print(f"Loading and compiling the model {path}")
    with session.create(str(path)) as sess:
        print(f"Model has been compiled successfully")

        print("Model input and output:")
        print(sess.print_summary())

        _, height, width, channel = sess.input(0).shape()
        image = image.convert('RGB').resize((width, height))

        data = np.zeros((width, height, channel), np.uint8)
        data[:width, :height, :channel] = image
        data = np.reshape(data, (1, width, height, channel))

        start_time = time.time()
        outputs = sess.run(data)
        print('Prediction elapsed {:.2f} secs'.format(time.time() - start_time))

        classified = np.squeeze(outputs[0].numpy())
        imagenet_labels = load_labels(current_path.joinpath('../assets/labels/ImageNetLabels.txt'))
        Class = collections.namedtuple('Class', ['id', 'score'])
        objects = []
        for idx, n in np.ndenumerate(classified):
            objects.append(Class(idx[0], n))

        objects.sort(key=lambda x: x[1], reverse=True)
        print("[Top 5 scores:]")
        for object in objects[:5]:
            print("{}: {}".format(imagenet_labels[object.id], object.score))


if __name__ == "__main__":
    if len(sys.argv) != 2:
       sys.stderr.write("./classify.py <image>\n")
       sys.exit(-1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        sys.stderr.write(f"{image_path} not found")
        sys.exit(-1)

    classify(image_path)
