import json
import tempfile
import unittest

import numpy as np

from furiosa.runtime import session
from furiosa.runtime.profiler import profile
from tests.test_base import MNIST_TFLITE_QUANTIZED


class TestProfiler(unittest.TestCase):
    def test_profile(self):
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
            self.assertTrue(len([record for record in records if record["name"] == "Run"]) == 2)
