import json
import tempfile
import unittest

import numpy as np

from furiosa.runtime import session
from furiosa.runtime.profiler import RecordFormat, profile
from tests.test_base import MNIST_TFLITE_QUANTIZED


class TestProfiler(unittest.TestCase):
    def test_profile_chrome_trace(self):
        with tempfile.TemporaryFile() as f:
            # Record profile data into temporary file
            with profile(file=f) as profiler:
                with session.create(MNIST_TFLITE_QUANTIZED) as sess:
                    input_meta = sess.inputs()[0]

                    input = np.random.randint(0, 127, input_meta.shape, dtype=np.uint8)

                    with profiler.record("Run"):
                        sess.run(input)

            f.seek(0)

            records = json.loads(f.read())
            self.assertEqual(len([record for record in records if record["name"] == "Run"]), 1)

    def test_profile_data_frame(self):
        # Record profile data to Pandas DataFrame
        with profile(format=RecordFormat.PandasDataFrame) as profiler:
            with session.create(MNIST_TFLITE_QUANTIZED) as sess:
                input_meta = sess.inputs()[0]

                input = np.random.randint(0, 127, input_meta.shape, dtype=np.uint8)

                with profiler.record("Run"):
                    sess.run(input)

        # Test for DataFrame correctness
        df = profiler.get_pandas_dataframe()
        self.assertEqual(len(df[df["name"] == "Run"].index), 1)

        # Test for exporting to [Chrome Trace Format]
        with tempfile.NamedTemporaryFile() as f:
            profiler.export_chrome_trace(f.name)
            records = json.loads(f.read())
            self.assertEqual(len([record for record in records if record["name"] == "Run"]), 1)
