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

Registry sources. As long as there is a descriptor file called `artifact` that defines where to fetch the model binary and model descriptions, it does not matter what form the registry should be. Github repository is a typical source.

**aritfact.yaml or artifact.toml**

Model descriptor file. You can find a complete schema in `artfiact_schema.json` or `furiosa/registry/artifact.py`

For example,

```yaml
artifacts:
  - name: mlcommons_resnet50_v1.5_int8
    family: ResNet
    location: >-
      https://github.com/furiosa-ai/npu-models/blob/master/mlcommons/mlcommons_resnet50_v1.5_int8.onnx
    format: onnx
    description: ResNet v1.5 model for MLCommons
    config:
      npu_device: npu0pe0
      compiler_config:
        keep_unsignedness: true
        split_unit: 0
    metadata:
      arxiv: 'https://arxiv.org/abs/1512.03385.pdf'
  - name: mlcommons_ssd_mobilenet_v1_int8
    family: MobileNetV1
    location: >-
      https://github.com/furiosa-ai/npu-models/blob/master/mlcommons/mlcommons_ssd_mobilenet_v1_int8.onnx
    format: onnx
    description: MobileNet v1 model for MLCommons
    config:
      npu_device: npu0pe0
    metadata:
      arxiv: 'https://arxiv.org/abs/1704.04861.pdf'
      input_shapes:
        - 3
        - 224
        - 224
```

**S3, Disk, HTTP**

Data binary transport types. Each repository can store their model binary in several types of transports. `location` field in `artifact.yaml` will be used to decide the transport.


## Getting started

**Install**

```sh
pip install furiosa-registry
```

**Fetch models**

```python
import asyncio
from typing import List

from furiosa.registry import request, Model


# Repository where to load model from
version = "v1.0"
repository = f"https://github.com/furiosa-ai/npu-models/tree/{version}/mlcommons"

# Load available models from the repository.
# Note that request() is an async function so we have to run the function in eventloop.
models: List[Model] = asyncio.run(request(repository, version=version))

# Pick a model
model = next(iter(models))

# Access the model
print(model.name)
print(model.version)
print(model.description)
# model binary: model.model
```

## Development

**Generate artfiact JSON schema from pydantic model definition.**

When you changed artifact schema, you can generate modified schema via

```sh
python -c 'from furiosa.registry import Artifact;\
            print(Artifact.schema_json(indent=2), file=open("artifact_schema.json", "w"))'
```

See `artifact_schema.json`
