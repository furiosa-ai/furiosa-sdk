#!/usr/bin/env python

import numpy as np
from furiosa.runtime.profiler import RecordFormat, profile
from furiosa.runtime.sync import create_runner

# You can find 'examples' directory of the root of furiosa-sdk source tree
model_path = "examples/assets/quantized_models/imagenet_224x224_mobilenet_v1_uint8_quantization-aware-trained_dm_1.0_without_softmax.tflite"

with profile(format=RecordFormat.PandasDataFrame) as profiler:
    with create_runner(model_path) as runner:
        input_shape = runner.model.input(0).shape

        # pause profiling during warmup
        profiler.pause()

        for _ in range(0, 10):
            with profiler.record("warm up") as record:
                runner.run([np.uint8(np.random.rand(*input_shape))])

        # resume profiling
        profiler.resume()

        with profiler.record("trace") as record:
            runner.run([np.uint8(np.random.rand(*input_shape))])

df = profiler.get_pandas_dataframe()

assert len(df[df["name"] == "trace"]) == 1
assert len(df[df["name"] == "warm up"]) == 0
