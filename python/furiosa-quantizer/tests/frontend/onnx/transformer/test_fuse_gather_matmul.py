import numpy as np
import torch
import torch.nn as nn

from furiosa.quantizer.frontend.onnx.transformer.fuse_gather_matmul import FuseGatherMatMul
from tests.frontend.onnx.transformer import TestTransformer


class UnitTestModel(nn.Module):
    def __init__(self, num_embs, emb_dims, out_dims):
        super(UnitTestModel, self).__init__()
        self.emb = nn.Embedding(num_embs, emb_dims)
        self.matmul = nn.Linear(emb_dims, out_dims, bias=False)

    def forward(self, x):
        x = self.emb(x)
        x = self.matmul(x)
        return x


class TestFuseGatherMatMul(TestTransformer):
    def _make_test_model(self, torch_model, input_shapes, dtype):
        orig_model, trans_model = self.make_test_model(
            torch_model, FuseGatherMatMul(), input_shapes, dtype
        )
        return orig_model, trans_model

    def test_case1(self):
        num_embs = 10
        emb_dims = 128
        out_dims = 8
        input_shapes = [(3)]

        op_types = ['Gather']

        orig_model, trans_model = self._make_test_model(
            UnitTestModel(num_embs, emb_dims, out_dims), input_shapes, dtype=torch.int64
        )

        self.check_graph_node(trans_model, op_types)
        self.check_output_value(
            orig_model,
            trans_model,
            input_shapes,
            data=[np.random.default_rng().integers(0, num_embs, size=input_shapes[0])],
        )
        self.check_value_info(trans_model)
