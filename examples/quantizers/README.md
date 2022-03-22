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

FuriosaAI NPU supports int8|uint8 per-channel|per-layer quantized models.

```console
$ python3 quantize.py
```

Please check out [inference examples](../inferences/) to learn how to inference through this quantized model.
