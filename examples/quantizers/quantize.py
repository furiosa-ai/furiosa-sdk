#!/usr/bin/env python3

from pathlib import Path
import sys

import onnx
import torch
import torchvision
from torchvision import transforms

from furiosa.quantizer import post_training_quantize


def main():
    assets = Path(__file__).resolve().parents[1] / "assets"

    model = onnx.load_model(assets / "fp32_models" / "mnist.onnx")
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )

    calibration_dataset = torchvision.datasets.MNIST(
        "./", train=False, transform=preprocess, download=True
    )
    calibration_dataloader = torch.utils.data.DataLoader(calibration_dataset, batch_size=1)

    model_quantized = post_training_quantize(
        model, ([image.numpy()] for image, _ in calibration_dataloader)
    )

    with open("mnist_quantized.dfg", "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(main())
