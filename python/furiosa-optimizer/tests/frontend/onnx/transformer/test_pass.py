import unittest

import onnxoptimizer

from furiosa.optimizer.frontend.onnx.utils.inference_shape import __SKIPPED_PASSES__

# Allowed passes with following package versions
# onnx-simplifier   0.4.13
# onnxoptimizer     0.3.6
__ALLOWED_PASSES__ = set(
    [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_if_with_const_cond',
        'eliminate_nop_cast',
        'eliminate_nop_concat',
        'eliminate_nop_dropout',
        'eliminate_nop_expand',
        'eliminate_nop_flatten',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'eliminate_nop_reshape',
        'eliminate_nop_split',
        'eliminate_nop_transpose',
        'eliminate_shape_gather',
        'eliminate_shape_op',
        'eliminate_slice_after_shape',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'fuse_concat_into_reshape',
        'fuse_consecutive_concats',
        'fuse_consecutive_log_softmax',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        'fuse_pad_into_pool',
        'fuse_transpose_into_gemm',
        'nop',
    ]
)


class TestPass(unittest.TestCase):
    def test_onnx_optimizer_pass(self):
        passes = set(onnxoptimizer.get_fuse_and_elimination_passes()).difference(
            set(__SKIPPED_PASSES__)
        )
        added_passes = passes.difference(__ALLOWED_PASSES__)
        if len(added_passes) != 0:
            raise Exception(f'New onnx optimizer passes are found {added_passes}')
