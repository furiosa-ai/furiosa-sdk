#!/usr/bin/env python

import numpy as np
from furiosa.runtime.profiler import profile
from furiosa.runtime.sync import create_runner

# You can find 'examples' directory of the root of furiosa-sdk source tree
model_path = "examples/assets/quantized_models/imagenet_224x224_mobilenet_v1_uint8_quantization-aware-trained_dm_1.0_without_softmax.tflite"

with open("mobilenet_v1_trace.json", "w") as output:
    with profile(file=output) as profiler:
        with create_runner(model_path) as runner:
            input_shape = runner.model.input(0).shape

            with profiler.record("warm up") as record:
                for _ in range(0, 2):
                    runner.run([np.uint8(np.random.rand(*input_shape))])

            with profiler.record("trace") as record:
                for _ in range(0, 2):
                    runner.run([np.uint8(np.random.rand(*input_shape))])
