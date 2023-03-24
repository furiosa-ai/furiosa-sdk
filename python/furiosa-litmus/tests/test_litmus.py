import os
import subprocess
import unittest


class LitmusTests(unittest.TestCase):
    data_path = os.path.dirname(__file__) + '/../../../tests/data'
    test_model = data_path + '/test.onnx'
    mnist_model = data_path + '/mnist-8.onnx'
    tflite_model = data_path + '/MNISTnet_uint8_quant_without_softmax.tflite'

    def test_test_model(self):
        result = subprocess.run(['furiosa-litmus', self.test_model], capture_output=True)
        self.assertTrue(result.returncode == 0, result.stderr)

    def test_mnist_model(self):
        result = subprocess.run(['furiosa-litmus', self.mnist_model], capture_output=True)
        self.assertTrue(result.returncode == 0, result.stderr)

    def test_tflite_model(self):
        result = subprocess.run(['furiosa-litmus', self.tflite_model], capture_output=True)
        self.assertTrue(result.returncode != 0, result.stderr)
