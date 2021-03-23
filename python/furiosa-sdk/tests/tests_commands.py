import os
import subprocess
import unittest
import uuid

from furiosa.cli import argparser
from tests import test_data


class CommandTests(unittest.TestCase):
    mnist_model = test_data('MNISTnet_uint8_quant_without_softmax.tflite')
    test_onnx_model = test_data('test.onnx')
    test_dynamic_ranges = test_data('test_dynamic_ranges.json')

    compiler_config = test_data('compiler_config.yml')
    invalid_compiler_config = test_data('invalid_compiler_config.yml')

    def setUp(self):
        self.parser = argparser.create_argparser()

    def assert_file_created(self, path, keep: bool = False):
        self.assertTrue(os.path.isfile(path))
        if not keep:
            os.remove(path)

    def test_no_command(self):
        result = subprocess.run(['furiosa'], capture_output=True)
        self.assertIn('ERROR: Need command', str(result.stderr))

    def test_compile(self):
        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile', self.mnist_model,
                                 ], capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('output.enf has been generated', str(result.stdout))

    def test_compile_only_target_npu_spec(self):
        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile', self.mnist_model,
                                 ], capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('output.enf has been generated', str(result.stdout))

    def test_compile_with_compiler_config(self):
        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile',
                                 self.mnist_model,
                                 '--config', self.compiler_config
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('output.enf has been generated', str(result.stdout))

    def test_compile_with_target_ir(self):
        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile',
                                 self.mnist_model,
                                 '--config', self.compiler_config,
                                 '--target-ir', 'lir'
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('output.lir has been generated', str(result.stdout))

    def test_compile_with_specific_output(self):
        output_path = '/tmp/{}.lir'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile',
                                 self.mnist_model,
                                 '--config', self.compiler_config,
                                 '--target-ir', 'lir',
                                 '-o', output_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{} has been generated'.format(output_path), str(result.stdout))
        self.assert_file_created(output_path)

    def test_compile_with_reports(self):
        import uuid

        output_path = '/tmp/{}.lir'.format(uuid.uuid4())
        compiler_report_file = '/tmp/{}.txt'.format(uuid.uuid4())
        mem_alloc_report_file = '/tmp/{}.html'.format(uuid.uuid4())

        result = subprocess.run(['furiosa',
                                 '-d',
                                 '-v',
                                 'compile',
                                 self.mnist_model,
                                 '--config', self.compiler_config,
                                 '--compiler-report', compiler_report_file,
                                 '--mem-alloc-report', mem_alloc_report_file,
                                 '-o', output_path
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{} has been generated'.format(output_path), str(result.stdout))

        self.assert_file_created(output_path)
        self.assert_file_created(compiler_report_file)
        self.assert_file_created(mem_alloc_report_file)

    def test_compile_with_invalid_config(self):
        result = subprocess.run(['furiosa',
                                 'compile',
                                 self.mnist_model,
                                 '--config', self.invalid_compiler_config,
                                 ],
                                capture_output=True)
        self.assertTrue(result.returncode != 0)
        self.assertIn('ERROR: fail to compile', str(result.stderr))
        self.assertIn(
            "thread \'main\' panicked at \'cannot load compiler config from a file.",
            str(result.stderr))

    def test_perfeye(self):
        output_path = '/tmp/{}.html'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'perfeye',
                                 self.mnist_model,
                                 '--config', self.compiler_config,
                                 '-o', output_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{} has been generated'.format(output_path), str(result.stdout))
        self.assert_file_created(output_path)

    def test_perfeye_with_compiler_config(self):
        output_path = '/tmp/{}.html'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'perfeye',
                                 self.mnist_model,
                                 '--config', self.compiler_config,
                                 '-o', output_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{} has been generated'.format(output_path), str(result.stdout))
        self.assert_file_created(output_path)

    def test_perfeye_with_target_npu_spec(self):
        output_path = '/tmp/{}.html'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'perfeye',
                                 self.mnist_model,
                                 # '--target-npu-spec', self.target_npu_spec,
                                 '--config', self.compiler_config,
                                 '-o', output_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('{} has been generated'.format(output_path), str(result.stdout))
        self.assert_file_created(output_path)

    def test_optimize_and_quantize(self):
        optimized_model_path = '/tmp/{}.onnx'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'optimize',
                                 self.test_onnx_model,
                                 '-o', optimized_model_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)

        quantized_model_path = '/tmp/{}.onnx'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 'quantize',
                                 optimized_model_path,
                                 '-o', quantized_model_path,
                                 '--dynamic-ranges', self.test_dynamic_ranges,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)

        self.assert_file_created(optimized_model_path)
        self.assert_file_created(quantized_model_path)

    def test_build_calibration_model(self):
        output_path = '/tmp/{}.onnx'.format(uuid.uuid4())
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'build_calibration_model',
                                 self.test_onnx_model,
                                 '-o', output_path,
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assert_file_created(output_path)

    def test_toolchain_list(self):
        result = subprocess.run(['furiosa',
                                 '-v',
                                 'toolchain',
                                 'list'
                                 ],
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stdout)
