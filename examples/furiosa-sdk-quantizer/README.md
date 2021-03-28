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
$ ./quantize.py ../assets/fp32_models/MobileNetV2_10c_10d.onnx ./quantized.onnx
[version: 12
]
Calibration: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  6.86it/s]
Quantization: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:02<00:00, 43.65it/s]
```

Then you can run the quantized model on FuriosaAI NPU. See [FuriosaAI NPU Python SDK Runtime Examples](../furiosa-sdk-runtime)
