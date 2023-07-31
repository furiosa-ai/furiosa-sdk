#!/usr/bin/env python

import numpy as np
from furiosa.runtime.profiler import RecordFormat, profile
from furiosa.runtime.sync import create_runner

# You can find 'examples' directory of the root of furiosa-sdk source tree
model_path = "examples/assets/quantized_models/imagenet_224x224_mobilenet_v1_uint8_quantization-aware-trained_dm_1.0_without_softmax.tflite"

with profile(format=RecordFormat.PandasDataFrame) as profiler:
    with create_runner(model_path) as runner:
        input_shape = runner.model.input(0).shape

        with profiler.record("warm up") as record:
            for _ in range(0, 2):
                runner.run([np.uint8(np.random.rand(*input_shape))])

        with profiler.record("trace") as record:
            for _ in range(0, 2):
                runner.run([np.uint8(np.random.rand(*input_shape))])

profiler.print_summary()  # (1)

profiler.print_inferences()  # (2)

profiler.print_npu_executions()  # (3)

profiler.print_npu_operators()  # (4)

profiler.print_external_operators()  # (5)

df = profiler.get_pandas_dataframe()  # (6)
print(df[df["name"] == "trace"][["trace_id", "name", "thread.id", "dur"]])
