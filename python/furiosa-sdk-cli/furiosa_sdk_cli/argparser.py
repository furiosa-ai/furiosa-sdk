import argparse

from furiosa_sdk_cli import utils

def create_argparser():
    parser = argparse.ArgumentParser(description='Furiosa AI Web Service CLI')
    parser.add_argument("-q", "--quiet", action="store_true",
                        help='Quiet mode, CLI will not print out any message', )
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Dnable debug mode")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")

    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser("version", help='Print out the version')

    compile_cmd = subparsers.add_parser("compile", help='Compile your model and generate a binary for Furiosa NPU')
    compile_cmd.add_argument('source', type=str,
                             help='Path to Model file (tflite, onnx, other renegade internal formats are supported)')
    compile_cmd.add_argument('-o', type=str,
                             help='Path to Output file')
    compile_cmd.add_argument('--target-ir', type=str, default='enf',
                             help='Target IR (available IRs: dfg, cdfg, gir, lir, enf)')
    compile_cmd.add_argument('--config', type=str,
                             help='Path to Compiler Config file (yaml)')
    compile_cmd.add_argument('--target-npu-spec', type=str,
                             help='Path to Target NPU Specification (yaml)')
    compile_cmd.add_argument('--compiler-report', type=str,
                             help='Path to the compiler report')
    compile_cmd.add_argument('--mem-alloc-report', type=str,
                             help='Path to the memory allocation report')

    perfeye_cmd = subparsers.add_parser("perfeye",
                                        help='Generate a visialized view of the static performance estimation')
    add_perf_opts(perfeye_cmd, 'html')

    optimize_cmd = subparsers.add_parser("optimize",
                                         help='Optimize a model to calibrate & quantize.')
    optimize_cmd.add_argument('source', type=str,
                              help='Path to onnx file')
    optimize_cmd.add_argument('-o', type=str, default='output.onnx',
                              help='Path to Output file (default: output.onnx)')

    build_calibration_model_cmd = subparsers.add_parser("build_calibration_model",
                                                        help='Build a model to a calibration model.')
    build_calibration_model_cmd.add_argument('source', type=str,
                                             help='Path to onnx file')
    build_calibration_model_cmd.add_argument('-o', type=str, default='output.onnx',
                                             help='Path to Output file (default: output.onnx)')

    quantize_cmd = subparsers.add_parser("quantize", help='Transform a model to a quantized model.')
    quantize_cmd.add_argument('source', type=str,
                              help='Path to onnx file')
    quantize_cmd.add_argument('-o', type=str, default='output.onnx',
                              help='Path to Output file (default: output.onnx)')
    quantize_cmd.add_argument('--dynamic-ranges', type=str,
                              help='path of the dynamic ranges')

    toolchain_cmd = subparsers.add_parser("toolchain", help='Compile your model and generate a binary for Furiosa NPU')
    toolchain_subcmd = toolchain_cmd.add_subparsers(dest="subcmd")
    toolchain_subcmd.add_parser("list", help='List all toolchains')

    validate = utils.which("furiosa-validate")
    if validate is not None:
        validate_cmd = subparsers.add_parser("validate", help='Validate an onnx model')
        validate_cmd.add_argument('model_path', type=str, help='Path to onnx model')
        validate_cmd.add_argument('-o', dest="output", type=str, default='validation.txt', help='Path to the validation output')

    return parser


def add_perf_opts(parser, content_type):
    parser.add_argument('source', type=str,
                        help='Path to Model file (tflite, onnx, other renegade internal formats are supported)')
    parser.add_argument('-o', type=str, default='output.{}'.format(content_type),
                        help='Path to Output file')
    parser.add_argument('--config', type=str,
                        help='Path to Compiler Config file (yaml)')
    parser.add_argument('--target-npu-spec', type=str,
                        help='Path to Target NPU Specification (yaml)')
