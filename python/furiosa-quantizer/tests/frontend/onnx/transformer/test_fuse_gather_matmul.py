import numpy as np

from furiosa.quantizer.frontend.onnx.transformer.fuse_gather_matmul import Pattern_1
from tests.frontend.onnx.transformer import TestTransformer


class TestFuseGatherMatMul(TestTransformer):
    def test_case1(self):
        in_dims = 3
        num_embs = 10
        emb_dims = 128
        out_dims = 8

        model_desc = {
            "input": {"x": (np.int64, [in_dims])},
            "output": {"y": (np.float32, [in_dims, out_dims])},
            "initializer": {
                "table": (np.float32, [num_embs, emb_dims]),
                "w": (np.float32, [emb_dims, out_dims]),
            },
            "node": [
                ("Gather", ["table", "x"], ["0"]),
                ("MatMul", ["0", "w"], ["y"]),
            ],
        }

        orig_model, trans_model = self.make_test_model(model_desc, Pattern_1)
        self.check_graph_node(trans_model, op_types=['Gather'])
        self.check_output_value(
            orig_model,
            trans_model,
            [(in_dims)],
            data=[np.random.default_rng().integers(0, num_embs, size=in_dims)],
        )
        self.check_value_info(trans_model)
