import onnx

from furiosa.quantizer.ir.common.operator import HorizontalPadding


def horizontal_pads(f1, f2, f3, f4, s1, s2, s3, s4):
    return [
        HorizontalPadding(f1, s1),
        HorizontalPadding(f2, s2),
        HorizontalPadding(f3, s3),
        HorizontalPadding(f4, s4),
    ]


def implicit_axis_to_explicit(axes, input_shape):
    assert len(input_shape) > 1

    if isinstance(axes, list):
        new_axes = []
        for axis in axes:
            if axis == -1:
                new_axes.append(len(input_shape) - 1)
            else:
                new_axes.append(axis)
        return new_axes
    elif isinstance(axes, int):
        axis = axes
        if axis == -1:
            return len(input_shape) - 1
        else:
            return axis
    else:
        raise Exception('Unknown type: %s. axes must be int or list.' % type(axes))


def gemm_shapes(input_shapes, transA, transB):
    if transA == 0:
        m, k = input_shapes[0]
    else:
        k, m = input_shapes[0]

    if transB == 0:
        k, n = input_shapes[1]
    else:
        n, k = input_shapes[1]

    return m, k, n


def slice_offset_dict(starts, axes, input_shape):
    offsets = [0] * len(input_shape)
    for start, axis in zip(starts, axes):
        offsets[axis] = start

    return offsets


def node_identifier(node: onnx.NodeProto) -> str:
    """
    In the case of onnx, FuriosaAI uses the first output node's name as a node identifier.
    """
    return node.output[0]
