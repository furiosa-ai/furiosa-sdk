Furiosa Serving
-------------
Furiosa serving is a lightweight library based on [FastAPI](https://fastapi.tiangolo.com/) to make a model server running on a Furiosa NPU.

## Dependency
Furiosa serving depends on followings:

- Furiosa NPU
- [furiosa-server](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server)
- [furiosa-registry](https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-registry)

## Installation
`furiosa-serving` can be installed from PyPI using `pip` (note that the package name is different from the importable name)::

```sh
pip install 'furiosa-sdk[serving]'
```

## Getting started
There is one main API called `ServeAPI`. You can think of `ServeAPI` as a kind of `FastAPI` wrapper.


### Run server

```python
# main.py
from fastapi import FastAPI
from furiosa.serving import ServeAPI

serve = ServeAPI()

# This is FastAPI instance
app: FastAPI = serve.app
```

You can run [uvicorn](https://www.uvicorn.org/) server via internal `app` variable from `ServeAPI` instance like [normal FastAPI application](https://fastapi.tiangolo.com/tutorial/first-steps/#first-steps)

```sh
$ uvicorn main:app # or uvicorn main:serve.app
```

### Load model

From `ServeAPI`, you can load your model binary which will be running on a Furiosa NPU. You should specify model name and URI where to load the model. URI can be one of them below

- Local file
- HTTP
- [S3](https://docs.aws.amazon.com/s3/index.html)

Note that model binary which is now supported by Furiosa NPU should be one of them below

- [Tensorflow lite (tflite)](https://www.tensorflow.org/lite/)
- [ONNX](https://github.com/onnx/onnx)

```python
from furiosa.common.thread import synchronous
from furiosa.serving import ServeAPI, ServeModel


serve = ServeAPI()


# Load model from local disk
imagenet: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
    location='./examples/assets/models/image_classification.onnx'
)

# Load model from HTTP
resnet: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
     location='https://raw.githubusercontent.com/onnx/models/main/vision/classification/resnet/model/resnet50-v1-12.onnx'
)

# Load model from S3 (Auth environment variable for aioboto library required)
densenet: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
     location='s3://furiosa/models/93d63f654f0f192cc4ff5691be60fb9379e9d7fd'
)
```

### Define API

From a model you just created, you can define [FastAPI path operation decorator](https://fastapi.tiangolo.com/tutorial/first-steps/#define-a-path-operation-decorator) like `post()`, `get()` to expose API endpoints.

You should follow [FastAPI Request Body concept](https://fastapi.tiangolo.com/tutorial/body/) to correctly define payload.

> :warning: This example below is not actually working as you have to define your own preprocess(), postprocess() functions first.

```python
from typing import Dict

from fastapi import File, UploadFile
from furiosa.common.thread import synchronous
from furiosa.serving import ServeAPI, ServeModel
import numpy as np


serve = ServeAPI()


model: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
    location='./examples/assets/models/image_classification.onnx'
)

@model.post("/models/imagenet/infer")
async def infer(image: UploadFile = File(...)) -> Dict:
    # Convert image to Numpy array with your preprocess() function
    tensors: List[np.ndarray] = preprocess(image)

    # Infer from ServeModel
    result: List[np.ndarray] = await model.predict(tensors)

    # Classify model from numpy array with your postprocess() function
    response: Dict = postprocess(result)

    return response
```

After running uvicorn server, you can find [documentations](https://fastapi.tiangolo.com/#interactive-api-docs) provided by FastAPI at localhost:8000/docs


### Use sub applications

Furiosa serving provides predefined [FastAPI sub applications](https://fastapi.tiangolo.com/advanced/sub-applications/) to give you additional functionalities out of box.

You can mount the _sub applications_ using `mount()`. We provides several _sub applications_ like below

- **Repository**: model repository to list models and load/unload a model dynamically
- **Model**: model metadata, model readiness
- **Health**: server health, server readiness

```python
from fastapi import FastAPI
from furiosa.serving import ServeAPI
from furiosa.serving.apps import health, model, repository


# Create ServeAPI with Repository instance. This repository maintains models
serve = ServeAPI(repository.repository)

app: FastAPI = serve.app

app.mount("/repository", repository.app)
app.mount("/models", model.app)
app.mount("/health", health.app)
```

You can also find documentations for the _sub applications_ at `localhost:8000/{application}/docs`. Note that `model` _sub application_ has different default doc API like `localhost:8000/{application}/api/docs` since default doc URL conflicts model API.

### Use processors for pre/post processing

Furiosa serving provides several _processors_ which are predefined pre/post process functions to convert your data for each model.

You can directly use the `preprocess()`, `postprocess()` from `Processor` instance or use the `Processor` in the form of decorator. When used as a decorator, `Processor` call `preprocess()` and `postprocess()` before and after your function respectively.

```python
import numpy as np
from furiosa.common.thread import synchronous
from furiosa.serving import ServeModel, ServeAPI
from furiosa.serving.processors import ImageNet


serve = ServeAPI()

model: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
    location='./examples/assets/models/image_classification.onnx'
)

@model.post("/models/imagenet/infer")
@ImageNet(model=model, label='./examples/assets/labels/ImageNetLabels.txt')  # This makes infer() Callable[[UploadFile], Dict]
async def infer(tensor: np.ndarray) -> np.ndarray:
    return await model.predict(tensor)
```

For better understanding, this approximately describes how `infer()` function works internally

```python
# Create processor
processor = ImageNet(model=model, label='./examples/assets/labels/ImageNetLabels.txt')

# API endpoint signature replaced with ImageNet.preprocess()
def infer(image: PIL.image) -> Dict:

    # Preprocess image from API client from processor
    tensor: np.ndarray = processor.preprocess(image)

    # Call your function from tensor above
    output: np.ndarray = infer(tensor)

    # Postprocess output above from processor
    response: Dict = processor.postprocess(output)

    # Return response in the form of Dict which is defined at ImageNet.postprocess()
    return response
```

Note that you **must** call _processor_ decorator first to pass correct function signature to FastAPI route decoartor which will be used argument validation.

```python
# Correct:
@model.post("/models/imagenet/infer")
@ImageNet(tensor=model.inputs[0], label='./examples/assets/labels/ImageNetLabels.txt')  # This makes infer() Callable[[UploadFile], Dict]
async def infer(tensor: np.ndarray) -> np.ndarray:
    ...

# Wrong:
@ImageNet(tensor=model.inputs[0], label='./examples/assets/labels/ImageNetLabels.txt')  # This makes infer() Callable[[UploadFile], Dict]
@model.post("/models/imagenet/infer")
async def infer(tensor: np.ndarray) -> np.ndarray:
    ...
```

### Compose models

You can composite multiple models using [FastAPI dependency injection](https://fastapi.tiangolo.com/tutorial/dependencies/).

> :warning: This example below is not actually working as there is no SegmentNet in processors yet

```python
from fastapi import Depends
from furiosa.common.thread import synchronous
from furiosa.serving import ServeModel, ServeAPI
from furiosa.serving.processors import ImageNet, SegmentNet


serve = ServeAPI()

imagenet: ServeModel = synchronous(serve.model("nux"))(
    'imagenet',
    location='./examples/assets/models/image_classification.onnx'
)

segmentnet: ServeModel = synchronous(serve.model("nux"))(
    'segmentnet',
    location='./examples/assets/models/image_segmentation.onnx'
)

# Note that no "imagenet.post()" here not to expose the endpoint
async def classify(image: UploadFile = File(...)) -> List[np.ndarray]:
    tensors: List[np.arrary] = ImageNet(tensor=imagenet.inputs[0]).preprocess(image)
    return await imagenet.predict(tensors)

@segmentnet.post("/models/composed/infer")
async def segment(tensors: List[np.ndarray] = Depends(classify)) -> Dict:
    tensors = await model.predict(tensors)
    return SegmentNet(tensor=segmentnet.inputs[0]).postprocess(tensors)
```

### Example 1

You can find a complete example at `examples/image_classify.py`

```sh
cd examples

examples$ python image_classify.py
INFO:furiosa_sdk_runtime._api.v1:loaded dynamic library /home/ys/Furiosa/compiler/npu-tools/target/x86_64-unknown-linux-gnu/debug/libnux.so (0.4.0-dev d1720b938)
INFO:     Started server process [984608]
INFO:uvicorn.error:Started server process [984608]
INFO:     Waiting for application startup.
INFO:uvicorn.error:Waiting for application startup.
[1/6] 🔍   Compiling from tflite to dfg
Done in 0.27935523s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 1079.9143s
▪▪▪▪▪ [2/3] Lowering...Done in 93.315895s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 45.07178s
Done in 1218.3285s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.002127793s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.096237786s
[5/6] 🔍   Compiling from gir to lir
Done in 0.03271749s
[6/6] 🔍   Compiling from lir to enf
Done in 0.48739022s
✨  Finished in 1219.4524s
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

You can find available API in http://localhost:8000/docs#/

Send image to classify a image from server you just launched.

```sh
examples$ curl -X 'POST' \
  'http://127.0.0.1:8000/imagenet/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/images/car.jpg'

```

### Example 2

In many user scenarios, for each request users may want to split a large image into a number of small images, and process all of them at a time.
In this use cases, using multiple devices will be able to boost the throughput, eventually leading to lower latency.
This example `examples/number_classify.py` shows how to implement this usecase with session pool and Python async/await/gather.

```sh
cd examples

examples$ python number_classify.py
INFO:     Started server process [57892]
INFO:     Waiting for application startup.
2022-10-28T05:36:42.468215Z  INFO nux::npu: Npu (npu0pe0-1) is being initialized
2022-10-28T05:36:42.473084Z  INFO nux: NuxInner create with pes: [PeId(0)]
2022-10-28T05:36:42.503103Z  INFO nux::npu: Npu (npu1pe0-1) is being initialized
2022-10-28T05:36:42.507724Z  INFO nux: NuxInner create with pes: [PeId(0)]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


```

You can find available API in http://localhost:8000/docs#/

Send image to classify a image from server you just launched.

```sh
examples$ curl -X 'POST' \
  'http://127.0.0.1:8000/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/images/1234567890.jpg'

```

## Code

The code and issue tracker are hosted on GitHub:\
https://github.com/furiosa-ai/furiosa-sdk

## Contributing

We welcome many types of contributions - bug reports, pull requests (code, infrastructure or documentation fixes). For more information about how to contribute to the project, see the ``CONTRIBUTING.md`` file in the repository.
