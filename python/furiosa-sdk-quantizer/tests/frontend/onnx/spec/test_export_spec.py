from typing import Dict

import unittest

import numpy as np
import torch
import torch.nn as nn

from tests import torch_to_onnx, make_test_model, make_test_model_with_init_val
from furiosa_sdk_quantizer.frontend.onnx.spec.export_spec import OnnxExportSpec
from furiosa_sdk_quantizer.ir.spec import Spec


class Test_get_inputs_for_gen_spec(unittest.TestCase):
    def evaluate_test(self,
                      expected_input_shapes,
                      expected_output_shapes,
                      expected_attributes,
                      expected_init_shapes=None):
        model, node = make_test_model(expected_input_shapes,
                                      expected_output_shapes,
                                      expected_attributes,
                                      expected_init_shapes)

        input_shapes, output_shapes, attributes = OnnxExportSpec(model).get_inputs_for_gen_spec(node)

        if expected_init_shapes:
            expected_input_shapes += expected_init_shapes
        self.assertEqual(expected_input_shapes, input_shapes)
        self.assertEqual(expected_output_shapes, output_shapes)
        self.assertEqual(expected_attributes, attributes)

    def test_case_1(self):
        input_shapes = [(8, 8)]
        output_shapes = [(12, 12)]
        init_shapes = [(16, 16, 16), ()]
        attributes = dict()
        self.evaluate_test(input_shapes, output_shapes, attributes, init_shapes)

    def test_case_2(self):
        input_shapes = [(8, 8), (10, 10, 10)]
        output_shapes = [(12, 12), (7, 7, 7, 7), (3, 3, 3)]
        attributes = {
            'attr1': 1,
            'attr2': [1, 2, 3],
            'attr3': 1.2339999675750732,
        }
        self.evaluate_test(input_shapes, output_shapes, attributes)


class Test_conv2d(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Conv', check_model=True)
        assert node.op_type == 'Conv'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['batch'], spec.option.batch)
        self.assertEqual(expected['input_channel'], spec.option.input_channel)
        self.assertEqual(expected['output_channel'], spec.option.output_channel)
        self.assertEqual((expected['height'], expected['width']), (spec.option.input.height, spec.option.input.width))
        self.assertEqual(expected['kernel_shape'], (spec.option.kernel.height, spec.option.kernel.width))
        self.assertEqual(expected.get('strides', (1, 1)), (spec.option.stride.height, spec.option.stride.width))
        self.assertEqual(expected.get('dilations', (1, 1)), (spec.option.dilation.height, spec.option.dilation.width))
        self.assertEqual(expected.get('group', 1), spec.option.groups)
        self.assertEqual(expected.get('pads', (0, 0, 0, 0)), (spec.option.padding_spec.Custom.top,
                                                              spec.option.padding_spec.Custom.bottom,
                                                              spec.option.padding_spec.Custom.left,
                                                              spec.option.padding_spec.Custom.right))

    def test_case_1(self):
        input_shapes = [(1, 3, 8, 12)]
        output_shapes = [(1, 3, 7, 11)]
        attributes = {'kernel_shape': (2, 2)}
        init_shapes = [(3, 8, 2, 2)]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).conv2d(node)

        kwargs = {
            'batch': 1,
            'input_channel': 3,
            'output_channel': 3,
            'height': 8,
            'width': 12,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(2, 32, 64, 48)]
        output_shapes = [(2, 128, 22, 6)]
        attributes = {
            'kernel_shape': (2, 3),
            'strides': (3, 8),
            'dilations': (4, 5),
            'group': 16,
            'pads': (2, 4, 2, 4)

        }
        init_shapes = [(128, 32, 2, 4)]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).conv2d(node)

        kwargs = {
            'batch': 2,
            'input_channel': 32,
            'output_channel': 128,
            'height': 64,
            'width': 48,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_convtranspose2d(Test_conv2d, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='ConvTranspose', check_model=True)
        assert node.op_type == 'ConvTranspose'

        return model, node

    def test_case_1(self):
        input_shapes = [(1, 3, 8, 12)]
        output_shapes = [(1, 8, 9, 13)]
        attributes = {'kernel_shape': (2, 2)}
        init_shapes = [(3, 8, 2, 2)]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).convtranspose2d(node)

        kwargs = {
            'batch': 1,
            'input_channel': 3,
            'output_channel': 8,
            'height': 8,
            'width': 12,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(2, 32, 64, 48)]
        output_shapes = [(2, 512, 190, 379)]
        attributes = {
            'kernel_shape': (2, 3),
            'strides': (3, 8),
            'dilations': (4, 5),
            'group': 16,
            'pads': (2, 4, 2, 4)

        }
        init_shapes = [(128, 32, 2, 4)]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).convtranspose2d(node)

        kwargs = {
            'batch': 2,
            'input_channel': 32,
            'output_channel': 512,
            'height': 64,
            'width': 48,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_maxpool2d(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='MaxPool', check_model=True)
        assert node.op_type == 'MaxPool'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['batch'], spec.option.batch)
        self.assertEqual((expected['height'], expected['width']), (spec.option.input.height, spec.option.input.width))
        self.assertEqual(expected['kernel_shape'], (spec.option.kernel.height, spec.option.kernel.width))
        self.assertEqual(expected.get('strides', (1, 1)),
                         (spec.option.stride.height, spec.option.stride.width))
        self.assertEqual(expected.get('dilations', (1, 1)), (spec.option.dilation.height, spec.option.dilation.width))
        self.assertEqual(expected.get('pads', (0, 0, 0, 0)), (spec.option.padding_spec.Custom.top,
                                                              spec.option.padding_spec.Custom.bottom,
                                                              spec.option.padding_spec.Custom.left,
                                                              spec.option.padding_spec.Custom.right))

    def test_case_1(self):
        input_shapes = [(1, 3, 8, 12)]
        output_shapes = [(1, 3, 7, 10)]
        attributes = {'kernel_shape': (2, 3)}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).maxpool2d(node)

        kwargs = {
            'batch': 1,
            'height': 8,
            'width': 12,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(2, 32, 64, 48)]
        output_shapes = [(2, 32, 22, 6)]
        attributes = {
            'kernel_shape': (2, 3),
            'dilations': (2, 2),
            'strides': (3, 8),
            'pads': (1, 2, 1, 2),
        }

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).maxpool2d(node)
        kwargs = {
            'batch': 2,
            'height': 64,
            'width': 48
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_avgpool2d(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='AveragePool', check_model=True)
        assert node.op_type == 'AveragePool'

        return model, node

    def make_test_model_1(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='GlobalAveragePool', check_model=True)
        assert node.op_type == 'GlobalAveragePool'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['batch'], spec.option.batch)
        self.assertEqual((expected['height'], expected['width']), (spec.option.input.height, spec.option.input.width))
        self.assertEqual(expected['kernel_shape'], (spec.option.kernel.height, spec.option.kernel.width))
        self.assertEqual(expected.get('strides', (1, 1)),
                         (spec.option.stride.height, spec.option.stride.width))
        self.assertEqual(expected.get('pads', (0, 0, 0, 0)), (spec.option.padding_spec.Custom.top,
                                                              spec.option.padding_spec.Custom.bottom,
                                                              spec.option.padding_spec.Custom.left,
                                                              spec.option.padding_spec.Custom.right))

    def test_case_1(self):
        input_shapes = [(1, 3, 8, 12)]
        output_shapes = [(1, 3, 7, 10)]
        attributes = {'kernel_shape': (2, 3)}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).avgpool2d(node)

        kwargs = {
            'batch': 1,
            'height': 8,
            'width': 12,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(2, 32, 64, 48)]
        output_shapes = [(2, 32, 22, 7)]
        attributes = {
            'kernel_shape': (2, 3),
            'strides': (3, 8),
            'pads': (1, 2, 1, 2),
        }

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).avgpool2d(node)
        kwargs = {
            'batch': 2,
            'height': 64,
            'width': 48
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_3(self):
        input_shapes = [(1, 3, 8, 12)]
        output_shapes = [(1, 3, 1, 1)]
        attributes = {}
        model, node = self.make_test_model_1(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).avgpool2d(node)

        kwargs = {
            'kernel_shape': (8, 12),
            'batch': 1,
            'height': 8,
            'width': 12,
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_4(self):
        input_shapes = [(2, 32, 64, 48)]
        output_shapes = [(2, 32, 1, 1)]
        attributes = {}

        model, node = self.make_test_model_1(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).avgpool2d(node)
        kwargs = {
            'kernel_shape': (64, 48),
            'batch': 2,
            'height': 64,
            'width': 48
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_gemm(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Gemm', check_model=True)
        assert node.op_type == 'Gemm'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected.get('alpha', 1.0), spec.option.alpha)
        self.assertEqual(expected.get('beta', 1.0), spec.option.beta)
        self.assertEqual(expected['m'], spec.option.m)
        self.assertEqual(expected['k'], spec.option.k)
        self.assertEqual(expected['n'], spec.option.n)

    def test_case_1(self):
        input_shapes = [(42, 64), (64, 32)]
        output_shapes = [(42, 32)]
        attributes = {'transA': 0}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).gemm(node)

        kwargs = {'m': 42, 'k': 64, 'n': 32}

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(42, 64), (32, 64)]
        output_shapes = [(42, 32)]
        attributes = {'transA': 0, 'transB': 1}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).gemm(node)

        kwargs = {'m': 42, 'k': 64, 'n': 32}

        self.evaluate_test(kwargs, spec)

    def test_case_3(self):
        input_shapes = [(64, 42), (32, 64)]
        output_shapes = [(42, 32)]
        attributes = {'transA': 1, 'transB': 1}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).gemm(node)

        kwargs = {'m': 42, 'k': 64, 'n': 32}

        self.evaluate_test(kwargs, spec)

    def test_case_4(self):
        input_shapes = [(64, 42)]
        output_shapes = [(42, 32)]
        attributes = {'transA': 1, 'transB': 1}
        init_shapes = [(32, 64)]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).gemm(node)

        kwargs = {'m': 42, 'k': 64, 'n': 32}

        self.evaluate_test(kwargs, spec)


class Test_matmul(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='MatMul', check_model=True)
        assert node.op_type == 'MatMul'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['lhs_shape'], spec.option.lhs_shape)
        self.assertEqual(expected['rhs_shape'], spec.option.rhs_shape)

    def test_case_1(self):
        input_shapes = [(1, 16, 128, 256), (1, 16, 256, 368)]
        output_shapes = [(1, 16, 128, 368)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).matmul(node)

        kwargs = {'lhs_shape': [1, 16, 128, 256],
                  'rhs_shape': [1, 16, 256, 368]}

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(32, 48)]
        output_shapes = [(32, 24)]
        init_shapes = [(48, 24)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_shapes)
        spec = OnnxExportSpec(model).matmul(node)

        kwargs = {'lhs_shape': [32, 48],
                  'rhs_shape': [48, 24]}

        self.evaluate_test(kwargs, spec)


class Test_depthtospace(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='DepthToSpace', check_model=True)
        assert node.op_type == 'DepthToSpace'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['batch'], spec.option.batch)
        self.assertEqual(expected['height'], spec.option.height)
        self.assertEqual(expected['width'], spec.option.width)
        self.assertEqual(expected['channel'], spec.option.channel)
        self.assertEqual(expected['blocksize'], spec.option.block_size)
        self.assertEqual(expected['mode'], spec.option.mode)

    def test_case_1(self):
        input_shapes = [(1, 1024, 44, 88)]
        output_shapes = [(1, 256, 88, 176)]
        attributes = {
            'blocksize': 2
        }
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).depthtospace(node)

        kwargs = {
            'batch': 1,
            'height': 44,
            'width': 88,
            'channel': 1024,
            'mode': 'DepthColumnRow'
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(1, 1024, 44, 88)]
        output_shapes = [(1, 256, 88, 176)]
        attributes = {
            'blocksize': 2,
            'mode': 'CRD'
        }
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).depthtospace(node)

        kwargs = {
            'batch': 1,
            'height': 44,
            'width': 88,
            'channel': 1024,
        }
        kwargs.update(attributes)
        kwargs['mode'] = 'ColumnRowDepth'

        self.evaluate_test(kwargs, spec)


class Test_resize(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values):
        model, node = make_test_model_with_init_val(input_shapes, output_shapes, attributes, init_values,
                                                    op_type='Resize', check_model=True)
        assert node.op_type == 'Resize'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['roi'], spec.option.roi)
        self.assertEqual(expected['scales'], spec.option.scales)
        self.assertEqual(expected['sizes'], spec.option.sizes)

    def test_case(self):
        input_shapes = [(1, 3, 224, 224)]
        output_shapes = [(1, 3, 112, 112)]
        attributes = {}
        init_values = [np.array([]), np.array([]), np.array([1, 3, 112, 112])]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_values)
        spec = OnnxExportSpec(model).resize(node)

        kwargs = {'shape': [1, 3, 224, 224],
                  'roi': [],
                  'scales': [],
                  'sizes': [1, 3, 112, 112]}

        self.evaluate_test(kwargs, spec)


class Test_add(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Add', check_model=True)
        assert node.op_type == 'Add'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)

    def test_case(self):
        input_shapes = [(16, 17, 18, 19), (16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).add(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_sub(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Sub', check_model=True)
        assert node.op_type == 'Sub'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19), (16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).sub(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_mul(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Mul', check_model=True)
        assert node.op_type == 'Mul'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19), (16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).mul(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_div(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Div', check_model=True)
        assert node.op_type == 'Div'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19), (16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).div(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_exp(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Exp', check_model=True)
        assert node.op_type == 'Exp'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).exp(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_sigmoid(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Sigmoid', check_model=True)
        assert node.op_type == 'Sigmoid'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).sigmoid(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_softplus(unittest.TestCase):
    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.input_shape)

    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Softplus', check_model=True)
        assert node.op_type == 'Softplus'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).softplus(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_gelu(Test_add, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Gelu', check_model=False)
        assert node.op_type == 'Gelu'

        return model, node

    def test_case(self):
        input_shapes = [(16, 17, 18, 19)]
        output_shapes = [(16, 17, 18, 19)]
        attributes = {}
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).gelu(node)

        kwargs = {'shape': [16, 17, 18, 19]}

        self.evaluate_test(kwargs, spec)


class Test_reduce_mean(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='ReduceMean', check_model=True)
        assert node.op_type == 'ReduceMean'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['axes'], spec.option.axes)

    def test_case(self):
        input_shapes = [(12, 22, 32)]
        output_shapes = [(12, 1, 1)]
        attributes = {'axes': [-1, 1]}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).reduce_mean(node)

        kwargs = {'shape': [12, 22, 32],
                  'axes': [2, 1]}
        self.evaluate_test(kwargs, spec)


class Test_reduce_sum(Test_reduce_mean, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='ReduceSum', check_model=True)
        assert node.op_type == 'ReduceSum'

        return model, node

    def test_case(self):
        input_shapes = [(12, 22, 32)]
        output_shapes = [(12, 1, 1)]
        attributes = {'axes': [-1, 1]}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).reduce_sum(node)

        kwargs = {'shape': [12, 22, 32],
                  'axes': [2, 1]}
        self.evaluate_test(kwargs, spec)


class Test_reduce_l2(Test_reduce_mean, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='ReduceL2', check_model=True)
        assert node.op_type == 'ReduceL2'

        return model, node

    def test_case(self):
        input_shapes = [(12, 22, 32)]
        output_shapes = [(12, 1, 1)]
        attributes = {'axes': [-1, 1]}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).reduce_l2(node)

        kwargs = {'shape': [12, 22, 32],
                  'axes': [2, 1]}
        self.evaluate_test(kwargs, spec)


class Test_squeeze(Test_reduce_mean, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Squeeze', check_model=True)
        assert node.op_type == 'Squeeze'

        return model, node

    def test_case(self):
        input_shapes = [(12, 22, 32)]
        output_shapes = [(12,)]
        attributes = {'axes': [-1, 1]}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).squeeze(node)

        kwargs = {'shape': [12, 22, 32],
                  'axes': [2, 1]}
        self.evaluate_test(kwargs, spec)


class Test_unsqueeze(Test_reduce_mean, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Unsqueeze', check_model=True)
        assert node.op_type == 'Unsqueeze'

        return model, node

    def test_case(self):
        input_shapes = [(12, 22, 32)]
        output_shapes = [(12, 1, 22, 32, 1)]
        attributes = {'axes': [-1, 1]}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).unsqueeze(node)

        kwargs = {'shape': [12, 22, 32],
                  'axes': [2, 1]}
        self.evaluate_test(kwargs, spec)


class Test_reshape(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values):
        model, node = make_test_model_with_init_val(input_shapes, output_shapes, attributes, init_values,
                                                    op_type='Reshape', check_model=True)
        assert node.op_type == 'Reshape'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['input_shape'], spec.option.input_shape)
        self.assertEqual(expected['output_shape'], spec.option.output_shape)

    def test_case(self):
        input_shapes = [(32, 16, 256, 256)]
        output_shapes = [(32, 64, 128, 128)]
        attributes = {}
        init_values = [np.array([32, -1, 128, 128])]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_values)
        spec = OnnxExportSpec(model).reshape(node)

        kwargs = {
            'input_shape': [32, 16, 256, 256],
            'output_shape': [32, 64, 128, 128]
        }

        self.evaluate_test(kwargs, spec)


class Test_expand(Test_reshape, unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values):
        # TODO fix Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
        #  when check_model=True
        model, node = make_test_model_with_init_val(input_shapes, output_shapes, attributes, init_values,
                                                    op_type='Expand', check_model=False)
        assert node.op_type == 'Expand'

        return model, node

    def test_case(self):
        input_shapes = [(32, 1, 12, 12)]
        output_shapes = [(32, 8, 12, 12)]
        attributes = {}
        init_values = [np.array([32, 8, 12, 12])]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_values)
        spec = OnnxExportSpec(model).expand(node)

        kwargs = {
            'input_shape': [32, 1, 12, 12],
            'output_shape': [32, 8, 12, 12]
        }

        self.evaluate_test(kwargs, spec)


class Test_concatenation(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Concat', check_model=True)
        assert node.op_type == 'Concat'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['tensors'], spec.option.tensors)
        self.assertEqual(expected['axis'], spec.option.axis)

    def test_case_1(self):
        input_shapes = [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
        output_shapes = [(1, 2, 9)]
        attributes = {'axis': -1}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).concatenation(node)

        kwargs = {'tensors': [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                  'axis': 2}

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(1, 2, 3), (1, 2, 3), (1, 2, 3)]
        output_shapes = [(3, 2, 3)]
        attributes = {'axis': 0}

        model, node = make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).concatenation(node)
        kwargs = {'tensors': [[1, 2, 3], [1, 2, 3], [1, 2, 3]]}
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_transpose(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Transpose', check_model=True)
        assert node.op_type == 'Transpose'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['permutation'], spec.option.permutation)

    def test_case(self):
        input_shapes = [(1, 2, 3, 4)]
        output_shapes = [(2, 1, 4, 3)]
        attributes = {'perm': (1, 0, -1, 2)}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).transpose(node)

        kwargs = {'shape': [1, 2, 3, 4],
                  'permutation': [1, 0, 3, 2]}

        self.evaluate_test(kwargs, spec)


class Test_slice(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values):
        model, node = make_test_model_with_init_val(input_shapes, output_shapes, attributes, init_values,
                                                    op_type='Slice', check_model=True)
        assert node.op_type == 'Slice'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['offset'], spec.option.offset)

    def test_case(self):
        input_shapes = [(1, 3, 224, 224)]
        output_shapes = [(1, 2, 224, 224)]
        attributes = {}
        init_values = [np.array([1]), np.array([]), np.array([1])]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_values)
        spec = OnnxExportSpec(model).slice(node)

        kwargs = {
            'shape': [1, 3, 224, 224],
            'offset': [0, 1, 0, 0]
        }

        self.evaluate_test(kwargs, spec)


class Test_flatten(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Flatten', check_model=True)
        assert node.op_type == 'Flatten'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['axis'], spec.option.axis)

    def test_case_1(self):
        input_shapes = [(16, 2, 3, 4, 5, 6, 7, 8)]
        output_shapes = [(1, 16 * 2 * 3 * 4 * 5 * 6 * 7 * 8)]
        attributes = {'axis': 0}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).flatten(node)

        kwargs = {'shape': [16, 2, 3, 4, 5, 6, 7, 8]}
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(8, 2, 3, 4, 5, 6, 7, 8)]
        output_shapes = [(8 * 2 * 3, 4 * 5 * 6 * 7 * 8)]
        attributes = {'axis': 3}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).flatten(node)

        kwargs = {'shape': [8, 2, 3, 4, 5, 6, 7, 8]}
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_3(self):
        input_shapes = [(1, 2, 3, 4, 5, 6, 7, 8)]
        output_shapes = [(1 * 2 * 3 * 4 * 5 * 6 * 7, 8)]
        attributes = {'axis': -1}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).flatten(node)

        kwargs = {
            'shape': [1, 2, 3, 4, 5, 6, 7, 8],
            'axis': 7}

        self.evaluate_test(kwargs, spec)


class Test_pad(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values):
        model, node = make_test_model_with_init_val(input_shapes, output_shapes, attributes, init_values,
                                                    op_type='Pad', check_model=True)
        assert node.op_type == 'Pad'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['pad'], [(p.first, p.second) for p in spec.option.pad])

    def test_case(self):
        input_shapes = [(1, 2, 3, 4)]
        output_shapes = [(7, 10, 13, 16)]
        attributes = dict()
        init_values = [np.array([1, 2, 3, 4, 5, 6, 7, 8])]
        model, node = self.make_test_model(input_shapes, output_shapes, attributes, init_values)
        spec = OnnxExportSpec(model).pad(node)

        kwargs = {
            'shape': [1, 2, 3, 4],
            'pad': [(1, 5), (2, 6), (3, 7), (4, 8)]
        }

        self.evaluate_test(kwargs, spec)


class Test_layer_norm(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='LayerNorm', check_model=False)
        assert node.op_type == 'LayerNorm'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['input_shape'], spec.option.input_shape)
        self.assertEqual(expected['eps'], spec.option.eps)

    def test_case(self):
        input_shapes = [(1, 32, 64, 64)]
        output_shapes = [(1, 32, 64, 64)]
        attributes = {'epsilon': 9.999999747378752e-05}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).layer_norm(node)

        kwargs = {'input_shape': [1, 32, 64, 64], 'eps': attributes['epsilon']}
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)


class Test_split(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Split', check_model=False)
        assert node.op_type == 'Split'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['shape'], spec.option.shape)
        self.assertEqual(expected['split'], spec.option.split)
        self.assertEqual(expected['axis'], spec.option.axis)

    def test_case(self):
        input_shapes = [(1, 284, 2)]
        output_shapes = [(1, 284, 1), (1, 284, 1)]
        attributes = {
            'split': (1, 1),
            'axis': -1
        }
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).split(node)

        kwargs = {
            'shape': [1, 284, 2],
            'split': [1, 1],
            'axis': 2
        }

        self.evaluate_test(kwargs, spec)


class Test_softmax(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_shapes=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_shapes,
                                      op_type='Softmax', check_model=False)
        assert node.op_type == 'Softmax'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['input_shape'], spec.option.input_shape)
        self.assertEqual(expected.get('beta', 1.0), spec.option.beta)
        self.assertEqual(expected['axis'], spec.option.axis)

    def test_case(self):
        input_shapes = [(1, 1001, 224)]
        output_shapes = [(1, 1001, 224)]
        attributes = {'axis': -1}

        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).softmax(node)

        kwargs = {
            'input_shape': [1, 1001, 224],
            'axis': 2
        }

        self.evaluate_test(kwargs, spec)


class Test_clip(unittest.TestCase):
    def make_test_model(self, input_shapes, output_shapes, attributes, init_values=None):
        model, node = make_test_model(input_shapes, output_shapes, attributes, init_values,
                                      op_type='Clip', check_model=False)
        assert node.op_type == 'Clip'

        return model, node

    def make_test_model_1(self, kwargs, input_shapes):
        class Clip(nn.Module):
            def __init__(self, kwargs):
                super(Clip, self).__init__()
                self.kwargs = kwargs

            def forward(self, x):
                return torch.clamp(x, **self.kwargs)

        model = torch_to_onnx(Clip(kwargs), input_shapes)
        node = model.graph.node[0]
        assert node.op_type == 'Clip'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['input_shape'], spec.option.input_shape)
        self.assertEqual(expected.get('min', None), spec.option.min)
        self.assertEqual(expected.get('max', None), spec.option.max)

    def test_case_1(self):
        input_shapes = [(1, 2, 3, 4)]
        output_shapes = [(1, 2, 3, 4)]
        attributes = {
            'min': 0.0,
            'max': 6.0
        }
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).clip(node)

        kwargs = {
            'input_shape': [1, 2, 3, 4],
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_2(self):
        input_shapes = [(1, 2, 3, 4)]
        output_shapes = [(1, 2, 3, 4)]
        attributes = {
            'max': 0.0,
        }
        model, node = self.make_test_model(input_shapes, output_shapes, attributes)
        spec = OnnxExportSpec(model).clip(node)

        kwargs = {
            'input_shape': [1, 2, 3, 4],
        }
        kwargs.update(attributes)

        self.evaluate_test(kwargs, spec)

    def test_case_3(self):
        input_shapes = [(1, 2, 3, 4)]
        kwargs = {'min': 0.0}
        model, node = self.make_test_model_1(kwargs, input_shapes)
        spec = OnnxExportSpec(model).clip(node)

        kwargs.update({
            'input_shape': [1, 2, 3, 4],
        })

        self.evaluate_test(kwargs, spec)

    def test_case_4(self):
        input_shapes = [(1, 2, 3, 4)]
        kwargs = {'max': 0.0}
        model, node = self.make_test_model_1(kwargs, input_shapes)
        spec = OnnxExportSpec(model).clip(node)

        kwargs.update({
            'input_shape': [1, 2, 3, 4],
        })

        self.evaluate_test(kwargs, spec)

    def test_case_5(self):
        input_shapes = [(1, 2, 3, 4)]
        kwargs = {
            'min': 0.0,
            'max': 6.0
        }
        model, node = self.make_test_model_1(kwargs, input_shapes)
        spec = OnnxExportSpec(model).clip(node)

        kwargs.update({
            'input_shape': [1, 2, 3, 4],
        })

        self.evaluate_test(kwargs, spec)


class Test_multi_node_lp_norm(unittest.TestCase):
    def make_test_model(self, kwargs, input_shapes):
        class LpNorm(nn.Module):
            def __init__(self, kwargs):
                super(LpNorm, self).__init__()
                self.kwargs = kwargs

            def forward(self, x):
                return torch.nn.functional.normalize(x, **self.kwargs)

        model = torch_to_onnx(LpNorm(kwargs), input_shapes)
        node = model.graph.node[-1]
        from furiosa_sdk_quantizer.frontend.onnx.utils.inference_shape import InferenceShape
        model = InferenceShape(model).inference_shape()

        assert node.op_type == 'Div'

        return model, node

    def evaluate_test(self, expected: Dict, spec: Spec):
        self.assertEqual(expected['input_shape'], spec.option.input_shape)
        self.assertEqual(expected['p'], spec.option.p)
        self.assertEqual(expected['axis'], spec.option.axis)

    def test_case(self):
        input_shapes = [(1, 2, 3, 4)]
        kwargs = {
            'dim': 1,
            'p': 2
        }
        model, node = self.make_test_model(kwargs, input_shapes)
        spec, _ = OnnxExportSpec(model).multi_node_lp_norm(node)

        kwargs.update({
            'input_shape': [1, 2, 3, 4],
            'axis': 1
        })

        self.evaluate_test(kwargs, spec)
