import numpy as np

from furiosa.optimizer.frontend.onnx.transformer.fuse_gather_matmul import Pattern_1
from furiosa.optimizer.frontend.onnx.transformer.utils import check_value_info


def test_case1(make_transformer_model, check_output_value, check_graph_node):
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
    orig_model, trans_model = make_transformer_model(model_desc, Pattern_1)

    check_graph_node(trans_model, op_types=["Gather"])
    check_output_value(
        orig_model,
        trans_model,
        [(in_dims)],
        data=[np.random.default_rng().integers(0, num_embs, size=in_dims)],
    )
    check_value_info(trans_model)


def test_case2(make_transformer_model, check_output_value, check_graph_node):
    seq_len = 2
    in_dims = 3
    num_embs = 10
    emb_dims = 128
    out_dims = 8

    model_desc = {
        "input": {
            "x": (np.int64, [seq_len, in_dims]),
        },
        "output": {"y": (np.float32, [seq_len, in_dims, out_dims])},
        "initializer": {
            "table": (np.float32, [num_embs, emb_dims]),
            "w": (np.float32, [emb_dims, out_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"]),
            ("MatMul", ["0", "w"], ["y"]),
        ],
    }
    orig_model, trans_model = make_transformer_model(model_desc, Pattern_1)

    check_graph_node(trans_model, op_types=["Gather"])
    check_output_value(
        orig_model,
        trans_model,
        [(seq_len, in_dims)],
        data=[np.random.default_rng().integers(0, num_embs, size=(seq_len, in_dims))],
    )
    check_value_info(trans_model)


def test_case3(make_transformer_model, check_output_value, check_graph_node):
    """
    Test RHS of Pattern_1 condition 7
    """
    in_dims = 3
    num_embs = 10
    emb_dims = 128
    out_dims = 8

    model_desc = {
        "input": {"x": (np.int64, [in_dims])},
        "output": {"y": (np.float32, [out_dims, in_dims])},
        "initializer": {
            "table": (np.float32, [emb_dims, num_embs]),
            "w": (np.float32, [out_dims, emb_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"], {"axis": 1}),
            ("MatMul", ["w", "0"], ["y"]),
        ],
    }
    orig_model, trans_model = make_transformer_model(model_desc, Pattern_1)

    check_graph_node(trans_model, op_types=["Gather"])
    check_output_value(
        orig_model,
        trans_model,
        [(in_dims)],
        data=[np.random.default_rng().integers(0, num_embs, size=in_dims)],
    )
    check_value_info(trans_model)


def test_case4(make_transformer_model, check_graph_node):
    """
    Test case which does not meet condition 1
    """
    in_dims = 3
    num_embs = 10
    emb_dims = 128
    out_dims = 8

    model_desc = {
        "input": {
            "x": (np.int64, [in_dims]),
            "w": (np.float32, [emb_dims, out_dims]),
        },
        "output": {"y": (np.float32, [in_dims, out_dims])},
        "initializer": {
            "table": (np.float32, [num_embs, emb_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"]),
            ("MatMul", ["0", "w"], ["y"]),
        ],
    }
    _, trans_model = make_transformer_model(model_desc, Pattern_1)

    # fusion does not apply
    check_graph_node(trans_model, op_types=["Gather", "MatMul"])
    check_value_info(trans_model)


def test_case5(make_transformer_model, check_graph_node):
    """
    Test case which does not meet condition 2
    """
    in_dims = 16
    num_embs = 8
    emb_dims = 256
    out_dims = 4

    model_desc = {
        "input": {
            "x": (np.int64, [in_dims]),
            "table": (np.float32, [num_embs, emb_dims]),
        },
        "output": {"y": (np.float32, [in_dims, out_dims])},
        "initializer": {
            "w": (np.float32, [emb_dims, out_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"]),
            ("MatMul", ["0", "w"], ["y"]),
        ],
    }
    _, trans_model = make_transformer_model(model_desc, Pattern_1)

    # fusion does not apply
    check_graph_node(trans_model, op_types=["Gather", "MatMul"])
    check_value_info(trans_model)


def test_case6(make_transformer_model, check_graph_node):
    """
    Test case which does not meet condition 3 & 5
    """
    in_dims = 16
    num_embs = 8
    emb_dims = 256
    out_dims = 4

    model_desc = {
        "input": {"x": (np.int64, [in_dims])},
        "output": {"y": (np.int32, [in_dims, out_dims])},
        "initializer": {
            "table": (np.int32, [num_embs, emb_dims]),
            "w": (np.int32, [emb_dims, out_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"]),
            ("MatMul", ["0", "w"], ["y"]),
        ],
    }
    _, trans_model = make_transformer_model(model_desc, Pattern_1)

    # fusion does not apply
    check_graph_node(trans_model, op_types=["Gather", "MatMul"])
    check_value_info(trans_model)


def test_case7(make_transformer_model, check_graph_node):
    """
    Test case which does not meet condition 4
    """
    seq_len = 3
    in_dims = 32
    out_dims = 4

    model_desc = {
        "input": {"x": (np.int64, [seq_len, in_dims])},
        "output": {"y": (np.int32, [seq_len, out_dims])},
        "initializer": {
            "table": (np.int32, [in_dims]),
            "w": (np.int32, [in_dims, out_dims]),
        },
        "node": [
            ("Gather", ["table", "x"], ["0"]),
            ("MatMul", ["0", "w"], ["y"]),
        ],
    }
    _, trans_model = make_transformer_model(model_desc, Pattern_1)

    # fusion does not apply
    check_graph_node(trans_model, op_types=["Gather", "MatMul"])
    check_value_info(trans_model)
