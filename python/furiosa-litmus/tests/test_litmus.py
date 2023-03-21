import os
import subprocess
import unittest


class LitmusTests(unittest.TestCase):
    mnist_model = os.path.dirname(__file__) + '/../../../tests/data/mnist-8.onnx'

    def test_litmus(self):
        result = subprocess.run(['furiosa', 'litmus', self.mnist_model], capture_output=True)
        self.assertTrue(result.returncode == 0, result.stderr)
