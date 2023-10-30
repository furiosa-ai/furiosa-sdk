#!/usr/bin/env python

import time
import numpy as np
import onnx
import torch
import torchvision
from torchvision import transforms
import tqdm

from furiosa.optimizer import optimize_model
from furiosa.quantizer import get_pure_input_names, quantize, Calibrator, CalibrationMethod, ModelEditor, TensorType
from furiosa.runtime import session
from furiosa.runtime.profiler import profile


torch_model = torchvision.models.resnet50(weights='DEFAULT')
torch_model = torch_model.eval()

dummy_input = (torch.randn(1, 3, 224, 224),)

torch.onnx.export(
    torch_model,  # PyTorch model to export
    dummy_input,  # model input
    "resnet50.onnx",  # where to save the exported ONNX model
    opset_version=13,  # the ONNX OpSet version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the ONNX model's input names
    output_names=["output"],  # the ONNX model's output names
)

onnx_model = onnx.load_model("resnet50.onnx")
onnx_model = optimize_model(onnx_model)

calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)
calibrator.collect_data([[torch.randn(1, 3, 224, 224).numpy()]])
ranges = calibrator.compute_range()

editor = ModelEditor(onnx_model)
input_tensor_name = get_pure_input_names(onnx_model)[0]

# Convert the input type to uint8
editor.convert_input_type(input_tensor_name, TensorType.UINT8)

graph = quantize(onnx_model, ranges)

with open("trace.json", "w") as trace:
    with profile(file=trace) as profiler:
        with session.create(graph) as session:
            image = torch.randint(256, (1, 3, 224, 224), dtype=torch.uint8)
            with profiler.record("pre"):
                image = image.numpy()
            with profiler.record("inf"):
                outputs = session.run(image)
            with profiler.record("post"):
                prediction = np.argmax(outputs[0].numpy(), axis=1)
