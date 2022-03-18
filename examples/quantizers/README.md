# Examples of FuriosaAI NPU Python SDK Quantizer

This contains examples of FuriosaAI NPU Python SDK Quantizer

## Preliminaries
* [Furiosa Python SDK Installation](https://furiosa-ai.github.io/docs/latest/en/installation/python-sdk.html) ([Korean](https://furiosa-ai.github.io/docs/latest/ko/software/python-sdk.html))

## Setup
```
git clone https://github.com/furiosa-ai/furiosa-sdk
cd furiosa-sdk/examples/quantizers/
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

Please checkout [inference examples](../inferences/) to learn how to inference through this quantized model.