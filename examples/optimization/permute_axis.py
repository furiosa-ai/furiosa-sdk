#!/usr/bin/env python

import time

import numpy as np
import onnx
import torch
import tqdm

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
from furiosa.runtime import session
from furiosa.runtime.profiler import profile


onnx_model = onnx.load_model("model_nhwc.onnx")
onnx_model = optimize_model(onnx_model)

calibrator = Calibrator(onnx_model, CalibrationMethod.MIN_MAX_ASYM)
calibrator.collect_data([[torch.randn(1, 512, 512, 3).numpy()]])
ranges = calibrator.compute_range()

graph = quantize(onnx_model, ranges)

compiler_config = { "permute_input": [[0, 3, 1, 2]] }

with open("trace.json", "w") as trace:
    with profile(file=trace) as profiler:
        with session.create(graph, compiler_config=compiler_config) as session:
            image = torch.randint(256, (1, 3, 512, 512), dtype=torch.uint8)
            with profiler.record("pre"):
                image = image.numpy()
            with profiler.record("inf"):
                outputs = session.run(image)
            with profiler.record("post"):
                prediction = outputs[0].numpy()