#!/usr/bin/env python3
"""Image segmentation example
FP32 model `deeplabv3 resnet50` is required
The model can be downloaded via command `gdown --id 1MjuG6mk13Bca3bXdEWQF6R9tBngBcJER`
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


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


def run_segmentation(image_path: str, model_path: str, is_fp32: bool) -> None:
    from furiosa.runtime import session

    height = width = 520

    # preprocessing input image
    input_image = Image.open(image_path)
    input_array = preprocess(input_image, (height, width))

    if is_fp32:
        # ONNX runtime inference with the given FP32 model
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(model_path)
        start_time = time.time()
        output_tensor = sess.run(['out'], input_feed={'input': np.expand_dims(input_array, axis=0)})[0]
        rgb = decode_segmap(output_tensor.squeeze())
    else:
        # Furiosa runtime inference with the quantized model
        with session.create(str(model_path)) as sess:
            print("Model has been compiled successfully")
            print("Model input and output:")
            print(sess.print_summary())
            start_time = time.time()
            output_tensor = sess.run(np.expand_dims(input_array, axis=0))
            output_tensor = output_tensor[0].numpy()
            np_array = np.squeeze(output_tensor)
            print(np_array)
            rgb = decode_segmap(np_array)

    print('Prediction elapsed {:.2f} secs'.format(time.time() - start_time))

    # show images
    output_image = Image.fromarray(rgb).resize(input_image.size)
    result_image = Image.new(mode='RGB', size=(input_image.width + output_image.width, input_image.height))
    mask = Image.new("L", input_image.size, 128)
    composite_image = Image.composite(input_image, output_image, mask)
    result_image.paste(composite_image)
    result_image.paste(output_image, (input_image.width, 0))
    result_image.save(f'{os.path.basename(image_path).split(".")[0]}_result.jpg')
    print(f'{os.path.basename(image_path).split(".")[0]}_result.jpg has been written.')


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Inference using the existing quantized model")
        current_path = Path(__file__).absolute().parent
        model_path = current_path.joinpath("../assets/quantized_models/deeplabv3_resnet50_argmax_int8.onnx")
        is_fp32 = False
    elif len(sys.argv) == 3:
        print("Inference using given FP32 model")
        model_path = Path(sys.argv[2])
        is_fp32 = True
    else:
        sys.stderr.write("python image_segmentation.py <image> <model: optional>\n")
        sys.exit(-1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        sys.stderr.write(f"{image_path} not found")
        sys.exit(-1)

    if not model_path.exists():
        sys.stderr.write(f"{model_path} not found")
        sys.exit(-1)

    run_segmentation(str(image_path.resolve()), str(model_path.resolve()), is_fp32)
