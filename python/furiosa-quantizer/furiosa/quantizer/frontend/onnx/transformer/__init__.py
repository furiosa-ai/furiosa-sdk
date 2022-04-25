from collections import ChainMap, Counter, OrderedDict, defaultdict
from typing import List, Optional, Sequence, Set

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import make_tensor, make_tensor_value_info

from furiosa.quantizer.frontend.onnx.transformer import utils
from furiosa.quantizer.frontend.onnx.utils.check_model import check_model


class ONNXTransformer:
    check_runnable = True

    def __init__(self, model):
        self.model = utils.name_nodes(model)
        self.producer_map = {
            node_output: node for node in model.graph.node for node_output in node.output
        }
        self.consumer_map = defaultdict(list)
        for node in self.model.graph.node:
            for tensor in node.input:
                self.consumer_map[tensor].append(node)
        self.optimizer_map = OrderedDict({node.name: node for node in model.graph.node})
        self.node_input_map = {node.name: node.input for node in model.graph.node}
        self.initializer_map = {init.name: init for init in model.graph.initializer}

        self.initializer_vi_map = {
            init.name: make_tensor_value_info(init.name, init.data_type, init.dims)
            for init in model.graph.initializer
        }

        self.value_info_map = {vi.name: vi for vi in model.graph.value_info}
        self.graph_input_map = {inp.name: inp for inp in model.graph.input}
        self.graph_output_map = {out.name: out for out in model.graph.output}
        self.value_info_all = {
            **self.initializer_vi_map,
            **self.value_info_map,
            **self.graph_input_map,
            **self.graph_output_map,
        }

        self.input_count_map = Counter()
        for node in model.graph.node:
            for tensor_name in node.input:
                self.input_count_map[tensor_name] += 1

    def transform(self):
        outputs = list(self.graph_output_map.keys())
        # To prevent traversing cyclic connections
        visited: Set[str] = set()
        visited_node: List[onnx.NodeProto] = []

        while len(outputs) > 0:
            output = outputs.pop(0)

            if output not in self.producer_map:
                continue

            node = self.producer_map[output]

            # prevent duplicate specs from being created from nodes that have multiple outputs like Split.
            if node in visited_node:
                continue
            inputs = self.pattern_matching(node)

            # Put predecessor of node to new outputs
            outputs += list(filter(lambda input: input not in visited, inputs))
            visited.update(inputs)
            visited_node.append(node)

        return self.build_optimized_model(self.model)

    def update_graph_fields(self, model):
        for field in ['initializer', 'input', 'output', 'value_info']:
            model.graph.ClearField(field)
            getattr(model.graph, field).extend(self.get_map_values(field))
        return model

    def build_optimized_model(self, model):
        model = self.update_graph_fields(model)
        new_nodes = []
        for member in self.get_map_values('node'):
            if isinstance(member, onnx.NodeProto):
                new_nodes.append(member)
            elif isinstance(member, list):
                new_nodes.extend(member)
            else:
                raise Exception(member)

        model = utils.rebuild_model(model, new_nodes)
        check_model(model, self.check_runnable)

        return model

    def make_int64_initializer(self, name, target_name):
        return make_tensor(
            name,
            onnx.TensorProto.INT64,
            (len(self.get_value_info_shape(target_name)),),
            self.get_value_info_shape(target_name),
        )

    def copy_value_info(self, name):
        if name in self.graph_input_map:
            return self.graph_input_map[name]
        if name in self.value_info_map:
            return self.value_info_map[name]
        raise Exception(f'{name} not found.')

    def get_value_info_shape(self, tensor_name: str) -> List[int]:
        return [
            dim.dim_value for dim in self.value_info_all[tensor_name].type.tensor_type.shape.dim
        ]

    def get_value_info_dtype(self, tensor_name: str) -> int:
        return self.value_info_all[tensor_name].type.tensor_type.elem_type

    def get_map_values(self, field):

        if any(field == word for word in ['input', 'output']):
            field_map = 'graph_' + field + '_map'
        elif field == 'node':
            field_map = 'optimizer_map'
        else:
            field_map = field + '_map'

        return utils.make_unhashables_unique(getattr(self, field_map).values())

    def get_initializer_array(self, node_input):
        if node_input not in self.initializer_map:
            return None
        return numpy_helper.to_array(self.initializer_map[node_input])

    def get_init_node_input(self, node):
        # FIXME new implementation gets node.input and returns list of input with initializer,
        # so that takes any number of node inputs.
        assert len(node.input) == 2
        return next(
            (tensor_name for tensor_name in node.input[:2] if tensor_name in self.initializer_map),
            None,
        )

    def get_data_node_input(self, node):
        data_node_input = None
        for node_input in node.input:
            if node_input in self.initializer_map:
                continue
            data_node_input = node_input

        return data_node_input

    def get_node_input_idx(self, node_input):
        assert len(self.consumer_map[node_input]) == 1
        return list(self.consumer_map[node_input][0].input).index(node_input)

    def find_next_node(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        next_nodes = []
        for v in self.optimizer_map.values():
            if not v:
                continue
            if isinstance(v, list):
                v = v[0]
            if not any(output == v_input for output in node.output for v_input in v.input):
                continue
            next_nodes.append(v)

        return next_nodes

    def find_prev_node(self, node_input: str) -> onnx.NodeProto:
        if node_input not in self.producer_map:
            return None

        return self.producer_map[node_input]

    def is_same_shape(self, input_1, input_2):
        if self.get_value_info_shape(input_1) != self.get_value_info_shape(input_2):
            return False
        return True

    def traverse_prev_node(self, producer_map_key: str, target_op_types: List[str]):
        prev_node = self.find_prev_node(producer_map_key)

        if not prev_node:
            return None

        if not utils.is_op_type(prev_node.op_type, target_op_types):
            return False

        return prev_node

    def update_single_optimizer_map(self, node: onnx.NodeProto, dest_name):
        self.optimizer_map[dest_name] = node

    def update_multiple_optimizer_map(self, nodes: List[onnx.NodeProto], dest_name):
        self.optimizer_map[dest_name] = nodes

    def update_single_value_info_map(self, value_info: onnx.ValueInfoProto):
        if value_info.name in self.graph_input_map:
            self.graph_input_map[value_info.name] = value_info
        else:
            self.value_info_map[value_info.name] = value_info

    def update_multiple_value_info_map(self, value_infos: List[onnx.ValueInfoProto]):
        for vi in value_infos:
            self.update_single_value_info_map(vi)

    def update_single_initializer_map(self, initializer: onnx.TensorProto):
        self.initializer_map[initializer.name] = initializer

    def update_multiple_initializer_map(self, initializers: List[onnx.TensorProto]):
        for init in initializers:
            if not init:
                continue
            self.update_single_initializer_map(init)

    def pop_single_optimizer_map(self, node: onnx.NodeProto):
        self.optimizer_map[node.name] = []

    def pop_multiple_optimizer_map(self, nodes: List[onnx.NodeProto]):
        for node in nodes:
            self.pop_single_optimizer_map(node)

    def pop_single_value_info_map(self, vi: onnx.NodeProto):
        self.value_info_map.pop(vi.name)
        if vi.name in self.graph_output_map:
            self.value_info_map.pop(vi.name)
        if vi.name in self.graph_input_map:
            self.value_info_map.pop(vi.name)

    def pop_multiple_value_info_map(self, vis: List[onnx.ValueInfoProto]):
        for vi in vis:
            self.pop_single_value_info_map(vi)

    def pop_single_initializer_map(self, init: onnx.TensorProto):
        self.initializer_map.pop(init.name)

    def pop_multiple_initializer_map(self, nodes: List[onnx.TensorProto]):
        for node in nodes:
            self.pop_single_initializer_map(node)

    def bridge_disconnected_nodes(
        self, node_0: onnx.NodeProto, next_nodes: List[onnx.NodeProto], new_input
    ):
        """
        For a graph changed, for example,
            before) prev --> node_1 --> node_0 --> next
            after) prev --> node_1 --> (   ) -/-> next

        This function bridges node_1 and next as follows:
            prev --> node_1 --> next
            by assigning next.input[y] = node_1.output[x]
        """
        for next_node in next_nodes:
            for idx, next_node_input in enumerate(next_node.input):
                for node_output in node_0.output:
                    if node_output != next_node_input:
                        continue
                    next_node.input[idx] = new_input
                self.update_single_optimizer_map(next_node, next_node.name)

        for node_output in node_0.output:
            for idx, output in enumerate(self.model.graph.output):
                if node_output != output.name:
                    continue
                self.graph_output_map[node_output] = self.copy_value_info(new_input)

    def transform_to_eliminate(self, nodes_to_remove: List[onnx.NodeProto], new_input):
        """
        This function eliminates designated nodes and bridges the previous and next nodes of them.

        For example, if [B, C] is given to be removed, then removes [B, C] in A - B - C - D and connects [A, D] to make A - D.
        """
        self.pop_multiple_optimizer_map(nodes_to_remove)
        self.bridge_disconnected_nodes(
            nodes_to_remove[-1], self.find_next_node(nodes_to_remove[-1]), new_input
        )

    def transform_to_fuse(
        self,
        nodes_to_remove: List[onnx.NodeProto],
        nodes_to_add: Optional[List[onnx.NodeProto]] = None,
        inits_to_add: Optional[List[onnx.TensorProto]] = None,
        vis_to_add: Optional[List[onnx.ValueInfoProto]] = None,
    ):
        # Pattern should be linear, and if pattern's last node has multiple outputs, they should be specified in the transformed node.
        assert len(nodes_to_remove[-1].output) == len(nodes_to_add[-1].output)

        # remove nodes after the last node with multiple output receiver(except for the last node).
        last_node_with_multiple_output_receiver = None
        for i, node in enumerate(reversed(nodes_to_remove)):
            if i == 0:
                continue
            for output in node.output:
                if self.input_count_map[output] > 1:
                    last_node_with_multiple_output_receiver = len(nodes_to_remove) - 1 - i
            if last_node_with_multiple_output_receiver is not None:
                break

        if last_node_with_multiple_output_receiver is not None:
            nodes_to_remove = nodes_to_remove[last_node_with_multiple_output_receiver + 1 :]

        self.pop_multiple_optimizer_map(nodes_to_remove)
        if nodes_to_add:
            self.update_multiple_optimizer_map(nodes_to_add, nodes_to_remove[0].name)
        if inits_to_add:
            self.update_multiple_initializer_map(inits_to_add)
        if vis_to_add:
            self.update_multiple_value_info_map(vis_to_add)

    def pattern_matching(self, base_node):
        raise NotImplementedError

    def pattern_matcher(self, node, pattern_to_match: List[str]):
        decoded_pattern = [p.split('/') for p in pattern_to_match]
        decoded_pattern.reverse()

        op_type_0 = decoded_pattern.pop(0)
        if not utils.is_op_type(node.op_type, op_type_0):
            return None

        matched_nodes = [node]
        while decoded_pattern:
            op_type_1 = decoded_pattern.pop(0)

            node_1 = None
            # TODO impl "multi-path search"
            for node_input in node.input:
                node_1 = self.traverse_prev_node(node_input, op_type_1)
                if node_1:
                    break

            if not node_1:
                return None
            node = node_1
            matched_nodes.append(node)

        matched_nodes.reverse()
        return matched_nodes
