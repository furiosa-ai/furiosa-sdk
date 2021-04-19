#!/usr/bin/env python3

"""A post-training quantization example.
"""

from pathlib import Path
import sys
from typing import Dict, List

from PIL import Image
import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet

import furiosa_sdk_quantizer


def main():
    assets = Path(__file__).resolve().parent.parent / "assets"

    # Loads MobileNetV2_10c_10d.onnx, which is a floating-point model
    # that we will quantize.
    model = onnx.load(assets / "fp32_models" / "MobileNetV2_10c_10d.onnx")

    # Prepares a calibration dataset. We should preprocess the image
    # dataset in the same way as we did when training the model.
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(256, Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.485, 0.456, 0.406)),
                std=torch.tensor((0.229, 0.224, 0.225)),
            ),
        ]
    )
    dataset = ImageNet(assets / "imagenet", "val", transform=imagenet_transform)
    # The shape of the model's input is [N, C, H, W] where N = 1.
    dataloader = DataLoader(dataset, batch_size=1)
    # The name of the model's input is `input.1`.
    calibration_dataset: List[Dict[str, np.ndarray]] = [
        {"input.1": x.numpy()} for x, _ in dataloader
    ]

    # Quantizes the model using the calibration dataset.
    quantized_model = furiosa_sdk_quantizer.post_training_quantize(
        model, calibration_dataset
    )

    # Saves the quantized model.
    onnx.save(quantized_model, "MobileNetV2_10c_10d-quantized.onnx")


if __name__ == "__main__":
    sys.exit(main())
