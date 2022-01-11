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
