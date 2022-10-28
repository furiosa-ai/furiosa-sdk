import logging
import os
from pathlib import Path
import random

import numpy as np
import onnxruntime

from furiosa.runtime import session
from tests import test_data

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)


MNIST_ONNX = test_data("mnist-8.onnx")
MNIST_TFLITE_QUANTIZED = test_data("MNISTnet_uint8_quant_without_softmax.tflite")
NAMED_TENSORS_ONNX = test_data("named_tensors.onnx")
QUANTIZED_CONV_TRUNCATED_TEST = test_data("quantized_i8_conv_truncated.onnx")


def assert_tensors_equal(expected, result):
    assert np.allclose(expected, result, atol=1.0), "{} was expected, but the result was {}".format(
        expected, result
    )


class SessionTester:
    def __init__(self, model_path, compiler_config=None):
        self.session = session.create(model=model_path, compiler_config=compiler_config)

    def close(self):
        self.session.close()


class AsyncSessionTester:
    def __init__(self, model_path):
        (self.session, self.queue) = session.create_async(model=Path(model_path))

    def close(self):
        self.queue.close()
        self.session.close()


class PredictionTester:
    def __init__(self, model_path):
        self.sess = onnxruntime.InferenceSession(model_path)

    def _run_nux(self, inputs: np.ndarray):
        pass

    def _run_onnxrt(self, inputs: np.ndarray):
        input_name = self.sess.get_inputs()[0].name
        return self.sess.run(None, {input_name: inputs})

    def assert_equals(self, inputs: np.ndarray):
        tf_results = self._run_onnxrt(inputs)
        nux_results = self._run_nux(inputs)

        assert_tensors_equal(tf_results, nux_results)


class BlockingPredictionTester(PredictionTester):
    def __init__(self, model_path):
        self.nux_sess = session.create(model=model_path)
        super().__init__(model_path)

    def _run_nux(self, inputs: np.ndarray):
        return self.nux_sess.run(inputs)[0].numpy()

    def close(self):
        self.nux_sess.close()


class AsyncPredictionTester(PredictionTester):
    def __init__(self, model_path):
        nux_sess, nux_queue = session.create_async(model=model_path)
        self.nux_sess = nux_sess
        self.nux_queue = nux_queue
        super().__init__(model_path)

    def _run_nux(self, inputs: np.ndarray) -> np.ndarray:
        key = random.randint(0, 100)
        self.nux_sess.submit(inputs, context={'key': key})
        _, outputs = self.nux_queue.recv()
        return outputs[0].numpy()

    def close(self):
        self.nux_queue.close()
        self.nux_sess.close()


def exist_char_dev(dev_path: str) -> bool:
    """
    Return True if a specified device exists, or False
    """
    import stat

    # NPU device is a character device
    return stat.S_ISCHR(os.stat(dev_path).st_mode)


def ensure_test_device() -> bool:
    """
    Return True if a NPU device for unit test is ready, or False.
    """
    device = os.getenv('NPU_DEVNAME')
    return device is not None and exist_char_dev(f"/dev/{device}")
