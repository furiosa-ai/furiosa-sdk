from typing import List, Dict

import numpy as np
import onnxruntime as ort
import onnx

from onnx import numpy_helper, shape_inference
from furiosa_sdk_quantizer.frontend.onnx.transformer import utils
from furiosa_sdk_quantizer.frontend.onnx.utils.check_model import check_model
from onnx.helper import make_tensor, make_tensor_value_info, TensorProto


class InferenceShape:
    def __init__(self, model: onnx.ModelProto) -> onnx.ModelProto:
        self.model = model
        self.initializer = {init.name: init for init in self.model.graph.initializer}
        self.initializer_key = self.initializer.keys()
        self.value_info = {vi.name: vi for vi in self.model.graph.value_info}
        self.nodes_by_input_name = {
            node_input: node for node in self.model.graph.node for node_input in node.input
        }
        self.nodes_by_output_name = {
            node_output: node for node in self.model.graph.node for node_output in node.output
        }

    def inference_shape(self) -> onnx.ModelProto:
        self.to_static_shape_graph()
        input_value_names = [inp.name for inp in self.model.graph.input]
        for init in self.model.graph.initializer:
            if init.name not in input_value_names:
                dims = numpy_helper.to_array(init).shape
                value_info = make_tensor_value_info(init.name, init.data_type, dims)
                self.model.graph.input.append(value_info)

        self.model = shape_inference.infer_shapes(self.model)
        self.analyze_constant_of_shape()

        return self.model

    def to_static_shape_graph(self):
        tensor_to_be_value_analyzed = list()
        dynamic_shape_nodes = self.get_dynamic_shape_nodes()

        tensor_to_be_value_analyzed.extend(
            self.get_value_analysis_nodes(
                removed_nodes=dynamic_shape_nodes,
                target_op_types=["Reshape", "Pad", "Resize", "Expand"],
                value_analysis_op_types=["Concat", "Cast", "Shape"],
                dtype=TensorProto.INT64,
                rank=1,
            )
        )

        # find input of broad-casting mul/div operators with Gather/Add node as its input
        scalar_nodes = self.get_scalar_nodes()
        tensor_to_be_value_analyzed.extend(
            self.get_value_analysis_nodes(
                removed_nodes=scalar_nodes,
                target_op_types=["Mul", "Div"],
                value_analysis_op_types=["Gather", "Add"],
                dtype=TensorProto.FLOAT,
                rank=0,
            )
        )

        if tensor_to_be_value_analyzed:
            print("Run model on ONNXRuntime for value analysis. It will take some time..")
            # assign value-analyzed shape to dynamic-shaping operator as its initializer
            self.assign_value_analyzed_shapes_to_initializer(
                value_dict=self.run_onnx_model(
                    self.model.SerializeToString(), tensor_to_be_value_analyzed
                )
            )

            new_nodes = list(
                filter(
                    lambda node: node not in dynamic_shape_nodes + scalar_nodes,
                    self.model.graph.node,
                )
            )

            # rebuild model graph without nodes in shaping subgraph
            self.model = utils.rebuild_model(self.model, new_nodes, renaming=False)
            check_model(self.model)

    def analyze_constant_of_shape(self):
        value_info = {vi.name: vi for vi in self.model.graph.value_info}
        traversal = list()
        tensor_to_be_value_analyzed = list()

        for node in self.model.graph.node:
            if node.op_type == "ConstantOfShape":
                vi = value_info[node.output[0]]
                dtype = vi.type.tensor_type.elem_type
                rank = len([dim for dim in vi.type.tensor_type.shape.dim])
                start_vertex = node.output[0]
                traversal.extend(self.depth_first_search(start_vertex, end_op_type="Shape"))
                tensor_to_be_value_analyzed.append(start_vertex)
                vi = make_tensor_value_info(node.output[0], dtype, ("",) * rank)
                self.model.graph.output.append(vi)

        new_nodes = list(filter(lambda node: node not in traversal, self.model.graph.node))

        if tensor_to_be_value_analyzed:
            self.assign_value_analyzed_shapes_to_initializer(
                value_dict=self.run_onnx_model(
                    self.model.SerializeToString(), tensor_to_be_value_analyzed
                )
            )

            # rebuild model graph without nodes in shaping subgraph
            self.model = utils.rebuild_model(self.model, new_nodes)
            check_model(self.model)
        self.model = shape_inference.infer_shapes(self.model)

    def get_value_analysis_nodes(
        self, removed_nodes, target_op_types, value_analysis_op_types, dtype, rank
    ):
        tensor_to_be_value_analyzed = list()
        for node in removed_nodes:
            if self.nodes_by_input_name[node.output[0]].op_type not in target_op_types:
                continue
            if node.op_type not in value_analysis_op_types:
                continue
            tensor_to_be_value_analyzed.append(node.output[0])
            vi = make_tensor_value_info(node.output[0], dtype, ("",) * rank)
            self.model.graph.output.append(vi)

        return tensor_to_be_value_analyzed

    def assign_value_analyzed_shapes_to_initializer(self, value_dict):
        new_tensor_protos = list()
        new_vi_protos = list()
        for k, v in value_dict.items():
            if v.dtype == "float32":
                proto_dtype = TensorProto.FLOAT
            elif v.dtype == "int64":
                proto_dtype = TensorProto.INT64
            else:
                raise Exception()

            vals = v.flatten().tolist()

            new_tensor_protos.append(
                make_tensor(name=k, data_type=proto_dtype, dims=v.shape, vals=vals)
            )
            new_vi_protos.append(
                make_tensor_value_info(name=k, elem_type=proto_dtype, shape=v.shape)
            )

        self.model.graph.initializer.extend(new_tensor_protos)
        self.model.graph.input.extend(new_vi_protos)

    def get_dynamic_shape_nodes(self):
        traversal = list()

        for idx, node in enumerate(self.model.graph.node):
            if node.op_type == "Reshape":
                start_vertex = node.input[1]
            elif node.op_type == "Pad":
                start_vertex = node.input[1]
            elif node.op_type == "Resize":
                # sizes(=node.input[3]) is optional according to ONNX operator spec
                try:
                    start_vertex = node.input[3]
                except IndexError:
                    continue
            elif node.op_type == "Expand":
                start_vertex = node.input[1]
            else:
                continue

            # apply dfs algorithm
            traversal.extend(self.depth_first_search(start_vertex, end_op_type="Shape"))

        return traversal

    def get_scalar_nodes(self):
        traversal = list()

        for idx, node in enumerate(self.model.graph.node):
            if node.op_type != "Mul" and node.op_type != "Div":
                continue

            for node_input in node.input:
                if node_input in self.initializer_key:
                    continue

                try:
                    prev_node = self.nodes_by_output_name[node_input]
                except KeyError:

                    continue
                start_vertex = node_input

                if prev_node.op_type != "Gather" and prev_node.op_type != "Add":
                    continue
                start_vertex = node_input

                for prev_node_input in prev_node.input:
                    if prev_node_input in self.initializer_key:
                        continue

                    try:
                        prevprev_node = self.nodes_by_output_name[prev_node_input]
                    except KeyError:
                        continue

                    if prevprev_node.op_type != "Relu" and prevprev_node.op_type != "ReduceSum":
                        continue

                    # apply dfs algorithm
                    traversal.extend(self.depth_first_search(start_vertex, end_op_type="Relu"))

        return traversal

    def depth_first_search(self, start_vertex, end_op_type):
        traversal = list()
        visited = set()
        stack = [start_vertex]

        while stack:
            vertex = stack.pop()
            if vertex in self.initializer_key or start_vertex is None:
                continue
            node = self.nodes_by_output_name[vertex]
            if node.op_type != end_op_type:
                if vertex not in visited:
                    visited.add(vertex)
                    traversal.append(node)
                    stack.extend(reversed(node.input))
            else:
                traversal.append(node)

        return traversal

    @staticmethod
    def run_onnx_model(model: onnx.ModelProto, output_name: List[str]) -> Dict[str, np.ndarray]:
        """
        This function run onnx model on onnxruntime and get values for given output names.
        """

        # Log severity level 3(Error)
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(model)

        feed_dict = dict()

        for attr in sess.get_inputs():
            name = attr.name
            shape = attr.shape
            type = attr.type
            if type == "tensor(float)":
                dtype = np.float32
            elif type == "tensor(int64)":
                dtype = np.int64
            else:
                raise Exception("Unknown dtype: %s" % type)

            feed_dict[name] = np.ones(shape).astype(dtype)

        values = sess.run(output_name, feed_dict)

        return dict(zip(output_name, values))
