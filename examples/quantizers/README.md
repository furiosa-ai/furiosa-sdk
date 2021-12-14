# Examples of FuriosaAI NPU Python SDK Quantizer

This contains examples of FuriosaAI NPU Python SDK Quantizer

## Preliminaries
* [FuriosaAI NPU SDK insatllation](https://furiosa-ai.github.io/renegade-manual/sdk/latest/ko/installation/index.html)

## Setup
```
git clone https://github.com/furiosa-ai/furiosa-sdk
cd furiosa-sdk/examples/furiosa-sdk-quantizer
pip install -r requirements.txt
```

## Quantization examples

FuriosaAI NPU supports int8|uint8 per-channel|per-layer quantized models. The current version 
of FuriosaAI Quantizer only provides functionalities for performance evaluations.
Proper accuracy will be achieved via further releases.

```
$ ./quantize.py
Calibration: 100%|██████████████████████████████| 50/50 [00:02<00:00, 19.33it/s]
Quantization: 100%|███████████████████████████| 100/100 [00:01<00:00, 98.63it/s]
```

Then you can run the quantized model on FuriosaAI NPU. See [FuriosaAI NPU Python SDK Runtime Examples](../furiosa-sdk-runtime)

## An example of the model spec export

The following example allows users to print out some brief model skeleton metadata, such as operator type, and the shape of input and output tensors.

```sh
$ ./export_spec.py ../assets/fp32_models/MobileNetV2_10c_10d.onnx output.txt
[version: 12]

$ cat output.txt
...
- name: '537'
  option:
    Operator:
      Conv2d:
        input:
          height: 224
          width: 224
        kernel:
          height: 3
          width: 3
        stride:
          height: 2
          width: 2
        dilation:
          height: 1
          width: 1
        batch: 1
        input_channel: 3
        output_channel: 32
        groups: 1
        padding_spec:
          Custom:
            top: 1
            bottom: 1
            left: 1
            right: 1
```
