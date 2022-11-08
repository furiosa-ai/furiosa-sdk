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

    def test_compile_with_default_output_and_target_ir(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.ldfg"
            os.chdir(tmpdir)
            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '--target-ir', 'ldfg'], capture_output=True
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_output(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"
            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '-o', output_file], capture_output=True
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_other_outputs(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"
            analyze_memory_output = f"{tmpdir}/memory_analysis.html"
            dot_graph_output = f"{tmpdir}/graph.dot"
            result = subprocess.run(
                [
                    'furiosa-compile',
                    self.mnist_model,
                    '-o',
                    output_file,
                    '--dot-graph',
                    dot_graph_output,
                    '--analyze-memory',
                    analyze_memory_output,
                ],
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
            self.assert_file_created(dot_graph_output)
            self.assert_file_created(analyze_memory_output)
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
                    '--split-after-lower',
                    '--auto-batch-size',
                ],
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_genetic_optimization(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"

            result = subprocess.run(
                [
                    'furiosa-compile',
                    self.mnist_model,
                    '-o',
                    output_file,
                    '-ga',
                    'init_tactic=random,generation_limit=500',
                ],
                capture_output=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assert_file_created(output_file)
        finally:
            shutil.rmtree(tmpdir)

    def test_compile_with_genetic_optimization_errors(self):
        tmpdir = tempfile.mkdtemp()
        try:
            output_file = f"{tmpdir}/output.enf"

            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '-o', output_file, '-ga', 'init_tactic=abc'],
                capture_output=True,
            )
            self.assertTrue(result.returncode != 0, result.stderr)
            self.assertTrue(
                "ERROR: init_tactic must be either 'random' or 'heuristic'"
                in result.stderr.decode().strip()
            )

            result = subprocess.run(
                ['furiosa-compile', self.mnist_model, '-o', output_file, '-ga', 'abc=def'],
                capture_output=True,
            )
            self.assertTrue(result.returncode != 0, result.stderr)
            self.assertTrue(
                "ERROR: unknown genetic algorithm parameter: 'abc'"
                in result.stderr.decode().strip()
            )
        finally:
            shutil.rmtree(tmpdir)
