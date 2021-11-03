Furiosa Registry
----------------

Furiosa Registry defines `Model` and `Artifact` which allows client to communicate with several types of registries.

## Overview

![image](https://user-images.githubusercontent.com/87121185/134446310-a7dbcf66-0e0b-4f1b-9255-0b2213476c34.png)
**Client**

Client using Furiosa Registry library. It can be a model server to serve models provided by FuriosaAI.

**Furiosa Registry**

This library

**Registry A, B, C..**

Registry sources. As long as there is a descriptor file called `artifact` that defines where to list the model binary and model descriptions, it does not matter what form the registry should be. Github repository is a typical source.

**artifacts.yaml**

Model descriptor file. You can find a complete schema in `artfiact_schema.json` or `furiosa/registry/artifact.py`

For example,

```yaml
artifacts:
  - name: mlcommons_resnet50_v1.5_int8
    family: ResNet
    location: models/MNISTnet_uint8_quant_without_softmax.tflite
    format: tflite
    metadata:
      description: ResNet50 v1.5 model for MLCommons v1.1
      publication:
        url: https://arxiv.org/abs/1512.03385.pdf
  - name: mlcommons_ssd_mobilenet_v1_int8
    family: MobileNetV1
    location: https://github.com/furiosa-ai/furiosa-models/raw/main/models/mlcommons/mlcommons_ssd_mobilenet_v1_int8.onnx
    format: onnx
    description: MobileNet v1 model for MLCommons
    metadata:
      description: MobileNet v1 model for MLCommons v1.1
      publication:
        url: https://arxiv.org/abs/1704.04861.pdf
  - name: mlcommons_ssd_resnet34_int8
    family: ResNet
    location: models/model.py
    format: code
    metadata:
      description: ResNet34 model for MLCommons v1.1
      publication:
        url: https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
```

**S3, Disk, HTTP**

Data binary transport types. Each repository can store their model binary in several types of transports. `location` field in `artifacts.yaml` will be used to decide the transport.


## Getting started

**Install**

```sh
pip install furiosa-registry
```

**List up available models via artifact**

```python
import asyncio

import furiosa.registry as registry


# Repository where to load artifacts from.
repository = "https://github.com/furiosa-ai/furiosa-models"

# List up the available artifacts.
artifacts: List[registry.Artifact] = registry.list(repository)

# Access models from the artifacts
for artifact in arifacts:
    print(artifact.name)
```

**Load models**

```python
import asyncio

import furiosa.registry as registry


# Repository where to load model from.
repository = "https://github.com/furiosa-ai/furiosa-models"

# Model name described in 'artifacts.yaml' at the repository.
model = "mlcommons_resnet50"

# Model version described in 'artifacts.yaml' at the repository.
version = "v1.1

# Load available model from the repository.
# Note that request() is an async function so we have to run the function in eventloop.
models: registry.Model = asyncio.run(registry.load(uri=repository, name=model, version=version))

# Access the model
print(model.name)
print(model.version)
print(model.description)
```

**Print documentation about a model**

```python
import asyncio

import furiosa.registry as registry


# Repository where to load artifacts from.
repository = "https://github.com/furiosa-ai/furiosa-models"

# Model name described in 'artifacts.yaml' at the repository.
model = "mlcommons_resnet50"

# Render documentation provided by the repository for the  model.
print(registry.help(repository, model))
```

## Development

**Generate artfiact JSON schema from pydantic model definition.**

When you changed artifact schema, you can generate modified schema via

```sh
python -c 'from furiosa.registry import Artifact;\
            print(Artifact.schema_json(indent=2), file=open("artifact_schema.json", "w"))'
```

See `artifact_schema.json`
