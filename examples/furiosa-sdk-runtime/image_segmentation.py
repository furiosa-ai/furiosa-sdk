#!/usr/bin/env python3

"""Image segmentation example"""
from typing import Tuple

import sys

import numpy as np
import onnxruntime as ort

from PIL import Image
from pathlib import Path


def preprocess(img: Image.Image, size: Tuple[int, int]) -> np.array:
    img = img.resize(size)
    img_arr = np.asarray(img, dtype='float32')

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_arr /= 255.
    img_arr -= mean
    img_arr /= std

    img_arr = img_arr.transpose([2, 0, 1])

    return img_arr


def decode_segmap(image: np.array) -> np.array:
    # https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, len(label_colors)):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def run_segmentation(model_path: str, image_path: str) -> None:
    height = width = 520

    # preprocess
    input_image = Image.open(image_path)
    input_array = preprocess(input_image, (height, width))

    # NOTE: replace here with npu runtime
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(model_path)
    output_tensor = sess.run(['out'], input_feed={'input': np.expand_dims(input_array, axis=0)})[0]

    # decode
    rgb = decode_segmap(output_tensor.squeeze())

    # save
    result = Image.fromarray(rgb).resize(input_image.size)
    result.save(f'{image_path.split(".")[0]}_seg.jpg')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("./demo.py <model> <image>\n")
        sys.exit(-1)

    model_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    if not image_path.exists():
        sys.stderr.write(f"{image_path} not found")
        sys.exit(-1)

    if not model_path.exists():
        sys.stderr.write(f"{model_path} not found")
        sys.exit(-1)

    run_segmentation(str(model_path.resolve()), str(image_path.resolve()))
