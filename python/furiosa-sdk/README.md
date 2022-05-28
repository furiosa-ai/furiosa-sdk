# FuriosaAI SDK

Furiosa™ SDK is a software development kit (SDK) for running deep-neural network inference using FuriosaAI NPU chips. It is a collection of libraries and command line tools including compiler and profiler. It also provides Python bindings that allow users to develop their application easily with abundant Python ecosystem, such as NumPy, Jupyter notebooks and scientific Python packages.

## Documentation
* [Furiosa SDK Document (English)](https://furiosa-ai.github.io/docs/latest/en)
* [Furiosa SDK Document (Korean)](https://furiosa-ai.github.io/docs/latest/ko)

## Installation

You can install Furiosa SDK with pip.

```sh
pip install furiosa-sdk
```

You can install also furiosa-sdk with extra packages (e.g., litmus, quantizer):

```sh
pip install 'furiosa-sdk[litmus,quantizer]'
```

The following are the extra packages:
* models: Library which provides pre-tained models for Furiosa NPU
* litmus: Command line tool to check if a model is compatible with furiosa-sdk
* server: Serving framework enabling a DNN model to provide HTTP/GRPC endpoints
* serving: FastAPI-based Serving Library
* quantizer: Library which allows to quantize DNN models

## Releases
* [Furiosa SDK 0.6.1](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.6.1) (Latest)
* [Furiosa SDK 0.6.0](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.6.0)
* [Furiosa SDK 0.5.2](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.5.2)
* [Furiosa SDK 0.5.1](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.5.1)
* [Furiosa SDK 0.5.0](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.5.0)
* [Furiosa SDK 0.4.0](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.4.0)
* [Furiosa SDK 0.2.1](https://github.com/furiosa-ai/furiosa-sdk/releases/tag/v0.2.1)

## License

```
Copyright 2022 FuriosaAI, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
