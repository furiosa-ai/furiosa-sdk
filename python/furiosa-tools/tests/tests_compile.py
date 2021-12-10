import os
import shutil
import subprocess
import tempfile
import unittest

from tests import test_data


class CommandTests(unittest.TestCase):
    mnist_model = test_data('MNISTnet_uint8_quant_without_softmax.tflite')
    compiler_config = test_data('compiler_config.yml')
    invalid_compiler_config = test_data('invalid_compiler_config.yml')

    def assert_file_created(self, path, keep: bool = False):
        self.assertTrue(os.path.isfile(path))
        if not keep:
            os.remove(path)

    def test_version(self):
        result = subprocess.run(['furiosa-compile', '--version'], capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_compile(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"
            os.chdir(tmpdir)
            result = subprocess.run(['furiosa-compile', self.mnist_model], capture_output=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_target_npus(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"

            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '-o', output_file, '--target-npu', 'warb'],
                capture_output=True,
            )
            self.assertTrue(result.returncode != 0, result.stderr)

            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '-o', output_file, '--target-npu', 'warboy'],
                capture_output=True,
            )
            self.assertTrue(result.returncode == 0, result.stderr)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_optimization(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"
            result = subprocess.run(
                [
                    'furiosa-compile',
                    self.mnist_model,
                    '-o',
                    output_file,
                    '--batch-size',
                    '2',
                    '--auto-batch-size',
                ],
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)
