import argparse
import onnx

from pathlib import Path
from furiosa_sdk_runtime import session
from furiosa_sdk_quantizer.frontend.onnx import post_training_quantization_with_random_calibration
from furiosa_sdk_quantizer.frontend.onnx.quantizer.utils import QuantizationMode

def validate(input_file: Path):
    output_file = "quantized_model.onnx"

    if not input_file.exists():
        print(f'input file {input_file} does not exist')

    # Try quantization on input models
    print(f'quantizing model {input_file.name}')
    quantized_model = post_training_quantization_with_random_calibration(model=onnx.load_model(input_file),
                                                                         per_channel=True,
                                                                         static=True,
                                                                         mode=QuantizationMode.dfg,
                                                                         num_data=10)
    onnx.save_model(quantized_model, output_file)
    session.create(model=output_file)


def main():
    parser = argparse.ArgumentParser(description='Get input file to validate')
    parser.add_argument('-i', '--input', required=True, dest='input', type=str, help='input model file')
    args = parser.parse_args()
    validate(Path(args.input))


if __name__ == '__main__':
    main()
