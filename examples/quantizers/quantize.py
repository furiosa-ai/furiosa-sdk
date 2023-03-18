#!/usr/bin/env python3

from pathlib import Path
import sys

import onnx
import torch
import torchvision
from torchvision import transforms
import tqdm

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod


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

    model = optimize_model(model)

    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX)

    for calibration_data, _ in tqdm.tqdm(calibration_dataloader, desc="Calibration", unit="images", mininterval=0.5):
        calibrator.collect_data([[calibration_data.numpy()]])

    ranges = calibrator.compute_range()
    
    model_quantized = quantize(model, ranges)

    with open("mnist_quantized.dfg", "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(main())
