import unittest

from furiosa.client import CompilerClient
from tests import test_data


class TestCompilerClient(unittest.TestCase):
    client = None

    def setUp(self) -> None:
        self.client = CompilerClient()

    def tearDown(self) -> None:
        pass

    def test_compile(self):
        with open(test_data('MNISTnet_uint8_quant_without_softmax.tflite'), 'rb') as file:
            compile_task = self.client.submit_compile(source=file)
            compile_task.wait_for_complete()
            self.assertEqual(compile_task.phase(), 'Succeeded')
            self.assertIsNotNone(compile_task.list_artifacts())
            self.assertIsNotNone(compile_task.get_ir())
            self.assertIsNotNone(compile_task.get_logs())
            self.assertIsNotNone(compile_task.get_compiler_report())
            self.assertIsNotNone(compile_task.get_dot_graph())
            self.assertIsNotNone(compile_task.get_memory_alloc_report())

    def test_compile_with_configs(self):
        with open(test_data('MNISTnet_uint8_quant_without_softmax.tflite'), 'rb') as file:
            compile_task = self.client.submit_compile(source=file,
                                                      compiler_config={'keep_unsignedness': True},
                                                      target_npu_spec={})
            compile_task.wait_for_complete()
            self.assertEqual(compile_task.phase(), 'Succeeded')


if __name__ == '__main__':
    unittest.main()
