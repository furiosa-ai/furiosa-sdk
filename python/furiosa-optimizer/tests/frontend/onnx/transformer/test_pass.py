import unittest

import onnxoptimizer

# Allowed passes with followed package versions
# onnx-simplifier   0.4.13
# onnxoptimizer     0.3.6
__ALLOWED_PASSES__ = set(
    [
        'eliminate_nop_pad',
        'fuse_consecutive_log_softmax',
        'eliminate_nop_reshape',
        'eliminate_if_with_const_cond',
        'eliminate_shape_gather',
        'eliminate_deadend',
        'eliminate_nop_flatten',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_concat_into_reshape',
        'nop',
        'eliminate_nop_dropout',
        'eliminate_nop_expand',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'eliminate_nop_concat',
        'eliminate_nop_transpose',
        'eliminate_shape_op',
        'fuse_transpose_into_gemm',
        'eliminate_nop_cast',
        'fuse_consecutive_squeezes',
        'eliminate_identity',
        'eliminate_nop_split',
        'eliminate_slice_after_shape',
        'fuse_pad_into_pool',
        'fuse_consecutive_concats',
        'fuse_matmul_add_bias_into_gemm',
        'eliminate_nop_monotone_argmax',
        'fuse_pad_into_conv',
        'fuse_consecutive_transposes',
    ]
)


class TestPass(unittest.TestCase):
    def test_onnx_optimizer_pass(self):
        passes = set(onnxoptimizer.get_fuse_and_elimination_passes()).difference(
            set(['fuse_bn_into_conv', 'eliminate_duplicate_initializer', 'fuse_add_bias_into_conv'])
        )
        added_passes = passes.difference(__ALLOWED_PASSES__)
        if len(added_passes) != 0:
            raise RuntimeError(f'New onnx optimizer passes are found {added_passes}') from None
