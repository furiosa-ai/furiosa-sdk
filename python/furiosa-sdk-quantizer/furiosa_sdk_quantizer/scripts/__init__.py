from typing import Optional, Text, List

import sys
import json
import argparse

import onnx

import quantizer.frontend.onnx
from quantizer.frontend.onnx.quantizer import quantizer


def main():
    args = parse_args()
    args_map = vars(args)
    if args.command == 'export_spec':
        export_spec(args_map['i'], args_map['o'])
    elif args.command == 'optimize':
        optimize(args_map['i'], args_map['o'])
    elif args.command == 'build_calibration_model':
        build_calibration_model(args_map['i'], args_map['o'])
    elif args.command == 'quantize':
        quantize(args_map['i'], args_map['o'], args_map['dynamic_ranges'])
    elif args.command == 'post_training_quantization_with_random_calibration':
        post_training_quantization_with_random_calibration(args_map['i'], args_map['o'], args_map['n'])
    elif args.command == 'calibrate_with_random':
        calibrate_with_random(args_map['i'], args_map['o'], args_map['n'])
    else:
        raise Exception(f"Unsupported command, {args.command}")


def parse_args():
    parser = argparse.ArgumentParser(description='Furiosa AI quantizer')

    subparsers = parser.add_subparsers(dest='command')

    export_spec_cmd = subparsers.add_parser("export_spec", help='export_spec help')
    export_spec_cmd.add_argument('-i', type=str, help='Path to Model file (tflite, onnx are supported)')
    export_spec_cmd.add_argument('-o', type=str, help='Path to Output file')

    build_calibration_model_cmd = subparsers.add_parser("build_calibration_model", help='build calibrate model help')
    build_calibration_model_cmd.add_argument('-i', type=str, help='Path to Model file (tflite, onnx are supported)')
    build_calibration_model_cmd.add_argument('-o', type=str, help='Path to Output file')

    optimize_cmd = subparsers.add_parser("optimize", help='optimize help')
    optimize_cmd.add_argument('-i', type=str, help='Path to Model file (tflite, onnx are supported)')
    optimize_cmd.add_argument('-o', type=str, help='Path to Output file')

    quantize_cmd = subparsers.add_parser("quantize", help='quantize help')
    quantize_cmd.add_argument('-i', type=str, help='Path to Model file (tflite, onnx are supported)')
    quantize_cmd.add_argument('-o', type=str, help='Path to Output file')
    quantize_cmd.add_argument('-d', '--dynamic-ranges', type=str, help='Dynamic ranges')

    post_training_quantization_with_random_calibration = subparsers.add_parser(
        "post_training_quantization_with_random_calibration",
        help='calibrate help',
    )
    post_training_quantization_with_random_calibration.add_argument('-i', type=str,
                                                                    help='Path to Model file (tflite, onnx are supported)')
    post_training_quantization_with_random_calibration.add_argument('-o', type=str,
                                                                    help='Path to Output file')
    post_training_quantization_with_random_calibration.add_argument('-n', type=int,
                                                                    help='The number of random data')

    calibrate_with_random_cmd = subparsers.add_parser("calibrate_with_random", help='Output: dynamic ranges')
    calibrate_with_random_cmd.add_argument('-i', type=str, help='Path to Model file (tflite, onnx are supported)')
    calibrate_with_random_cmd.add_argument('-o', type=str, help='Path to Output file')
    calibrate_with_random_cmd.add_argument('-n', type=int, help='The number of random data')

    return parser.parse_args()


def export_spec(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    if output is not None:
        with open(output, 'w') as writable:
            quantizer.frontend.onnx.export_spec(model, writable)
    else:
        quantizer.frontend.onnx.export_spec(model, sys.stdout)


def optimize(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    model = quantizer.frontend.onnx.optimize_model(model)
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def build_calibration_model(input: Optional[Text] = None, output: Optional[Text] = None):
    model = _read_model(input)
    model = quantizer.frontend.onnx.build_calibration_model(model)
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def quantize(input: Optional[Text] = None,
             output: Optional[Text] = None,
             dynamic_ranges: str = None):
    model = _read_model(input)
    with open(dynamic_ranges, 'r') as readable:
        dynamic_ranges = json.load(readable)
    model = quantizer.frontend.onnx.quantize(model,
                                       per_channel=True,
                                       static=True,
                                       mode=quantizer.frontend.onnx.quantizer.QuantizationMode.dfg,
                                       dynamic_ranges=dynamic_ranges)
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def _read_model(input: Optional[Text] = None) -> onnx.ModelProto:
    if input is not None:
        with open(input, 'rb') as readable:
            model = onnx.load_model(readable, onnx.helper.ModelProto)
    else:
        model = onnx.load_model(sys.stdin, onnx.helper.ModelProto)

    return model


def post_training_quantization_with_random_calibration(input: Optional[Text] = None,
                                                       output: Optional[Text] = None,
                                                       num_data: Optional[int] = None):
    model = _read_model(input)
    model = quantizer.frontend.onnx.post_training_quantization_with_random_calibration(
        model,
        static=True,
        per_channel=True,
        mode=quantizer.QuantizationMode.dfg,
        num_data=num_data,
    )
    if output is not None:
        onnx.save_model(model, output)
    else:
        onnx.save_model(model, sys.stdout)


def calibrate_with_random(input: Optional[Text] = None,
                          output: Optional[Text] = None,
                          num_data: Optional[int] = None):
    model = _read_model(input)
    dynamic_ranges = quantizer.frontend.onnx.calibrate_with_random(model, num_data)
    if output is not None:
        with open(output, 'w') as f:
            json.dump(dynamic_ranges, f, ensure_ascii=True, indent=2)
    else:
        json.dump(dynamic_ranges, sys.stdout, ensure_ascii=True, indent=2)
