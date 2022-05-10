import json
import tempfile
import unittest

import numpy as np

from furiosa.runtime import session
from furiosa.runtime.profiler import profile
from tests.test_base import MNIST_TFLITE_QUANTIZED, test_data


class TestProfiler(unittest.TestCase):
    def test_profile(self):
        with tempfile.TemporaryFile() as f:
            # Record profile data into temporary file
            with profile(file=f.fileno()) as profiler:
                sess = session.create(MNIST_TFLITE_QUANTIZED)

                input_meta = sess.inputs()[0]

                input = np.random.randint(0, 127, input_meta.shape, dtype=np.uint8)

                with profiler.record("Run"):
                    sess.run(input)

            f.seek(0)

            with open(test_data("profiler_result.json")) as stub:
                expected = json.loads(stub.read())
                actual = json.loads(f.read())

                def records(records):
                    # Pair of (name, ph) is a identity of a record
                    return sorted((record["name"], record["ph"]) for record in records)

                self.assertEqual(records(expected), records(actual))
