import abc

from onnx import TensorProto
from onnx.helper import make_node, make_model, make_tensor_value_info
import torch
import torch.nn as nn

from furiosa_sdk_quantizer.frontend.onnx.transformer.deprecated.eliminate_identity import EliminateIdentity
from furiosa_sdk_quantizer.frontend.onnx.transformer.extract_constant_to_initializer import ExtractConstantToInitializer
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model

from tests import torch_to_onnx
from tests.frontend.onnx.transformer import TestTransformer


class TestEliminateIdentity(TestTransformer, abc.ABC):
    def test_case1(self):
        """
        Test case1
        from: input -> mul -> output
        to: input -> mul -> indentity -> output
        """

        class TestModel(nn.Module, abc.ABC):
            def forward(self, x):
                x = torch.mul(x, torch.ones(4))
                return x

        model = torch_to_onnx(TestModel(), [(1, 4)])

        new_nodes = []
        for node in model.graph.node:
            new_nodes.append(node)

        new_nodes.append(
            make_node(op_type='Identity',
                      inputs='2',
                      outputs='3')
        )
        model.graph.output.remove(model.graph.output[0])
        model.graph.output.append(
            make_tensor_value_info(name='3',
                                   elem_type=TensorProto.FLOAT,
                                   shape=(1, 4)))
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        model = make_model(model.graph)
        model = ExtractConstantToInitializer().transform(model)
        model = EliminateIdentity().transform(model)
        check_model(model)

        self.check_graph_node(model, ['Mul'])

    def test_case2(self):
        """
            Test case2
            from: input -> mul -> output
            to: input -> identity -> mul -> output
        """

        class TestModel(nn.Module, abc.ABC):
            def forward(self, x):
                x = torch.mul(x, torch.ones(4))
                return x

        model = torch_to_onnx(TestModel(), [(1, 4)])

        new_nodes = []
        for node in model.graph.node:
            if node.op_type != 'Mul':
                new_nodes.append(node)
                continue

            new_nodes.append(
                make_node(op_type='Identity',
                          inputs='0',
                          outputs='2')
            )
            new_nodes.append(node)
            node.input[0] = '2'
            node.output[0] = '3'

        model.graph.output[0].name = '3'
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        model = make_model(model.graph)
        model = ExtractConstantToInitializer().transform(model)
        model = EliminateIdentity().transform(model)
        check_model(model)

        self.check_graph_node(model, ['Mul'])

    def make_test_case3(self):
        """
            Test case2
            from: input -> mul -> output
            to: input -> mul -> identity -> output
                             -> identity -> output
        """

        class TestModel(nn.Module, abc.ABC):
            def forward(self, x):
                x = torch.mul(x, torch.ones(4))
                y = torch.add(x, torch.ones(4))
                z = torch.sub(x, torch.ones(4))
                return y, z

        model = torch_to_onnx(TestModel(), [(1, 4)])

        new_nodes = []
        outputs_by_name = {oup.name: oup for oup in model.graph.output}

        for node in model.graph.node:
            new_nodes.append(node)
            try:
                model.graph.output.remove(outputs_by_name[node.output[0]])
            except KeyError:
                continue

        new_nodes.append(
            make_node(op_type='Identity',
                      inputs='4',
                      outputs='7')
        )

        new_nodes.append(
            make_node(op_type='Identity',
                      inputs='6',
                      outputs='8')
        )

        model.graph.output.extend([
            make_tensor_value_info(name='7',
                                   elem_type=TensorProto.FLOAT,
                                   shape=(1, 4)),
            make_tensor_value_info(name='8',
                                   elem_type=TensorProto.FLOAT,
                                   shape=(1, 4))
        ])

        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)
        model = make_model(model.graph)
        model = ExtractConstantToInitializer().transform(model)
        model = EliminateIdentity().transform(model)
        check_model(model)

        self.check_graph_node(model, ['Mul', 'Add', 'Sub'])
