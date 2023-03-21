# Furiosa Model Server (Alpha)
Furiosa Model Server is a framework for serving Tflite/ONNX models through a REST API, using Furiosa NPUs.

Furiosa Model server API supoorts a REST and gRPC interface, compliant with [KFServing's V2
Dataplane](https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md)
specification and [Triton's Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md) specification.

## Features
- [x] HTTP REST API support
- [x] Multi-model support
- [x] GRPC support
- [x] OpenAPI specification support
- [ ] Compiler configuration support
- [ ] Input tensor adapter in Python (e.g., converting jpeg, png image files to tensors)
- [ ] Authentication support

# Building for Development
**Requirements**

* Python >= 3.8
* libnpu
* libnux

Install apt depdencies.
```sh
sudo apt install furiosa-libnpu-sim # or furiosa-libnpu-xrt if you have Furiosa H/W
sudo apt install furiosa-libnux
```

Install Python dependencies.

```sh
pip install -e .
```

To build from source, generate required files from [grpc tools](https://grpc.io/docs/languages/python/quickstart/) and [datamodel-codegen](https://koxudaxi.github.io/datamodel-code-generator/). Each step is needed to generate a GRPC stub and [pydantic](https://pydantic-docs.helpmanual.io/datamodel_code_generator/) data class.

**Generate GRPC API**
```sh
for api in "predict" "model_repository"
do
    python -m grpc_tools.protoc \
        -I"./proto" \
        --python_out="./furiosa/server/api/grpc/generated" \
        --grpc_python_out="./furiosa/server/api/grpc/generated" \
        --mypy_out="./furiosa/server/api/grpc/generated" \
        "./proto/$api.proto"
done
```

**Generate Pydantic data type**
```sh
for api in "predict" "model_repository"
do
    datamodel-codegen \
    --input "./openapi/$api.yaml" \
    --output "./furiosa/server/types/$api.py"
done
```

**Testing**

```sh
furiosa-server$ pytest --capture=no
============================================================ test session starts =============================================================
platform linux -- Python 3.9.6, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /home/ys/Furiosa/cloud/furiosa-server
plugins: asyncio-0.15.1
collected 10 items

tests/test_server.py [1/6] 🔍   Compiling from tflite to dfg
Done in 0.006840319s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 47.121174s
▪▪▪▪▪ [2/3] Lowering...Done in 19.422386s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 0.27680752s
Done in 66.82971s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.000951856s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.028555028s
[5/6] 🔍   Compiling from gir to lir
Done in 0.01069514s
[6/6] 🔍   Compiling from lir to enf
Done in 0.05054388s
✨  Finished in 66.980644s
.........[1/6] 🔍   Compiling from tflite to dfg
Done in 0.005259287s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 0.003461787s
▪▪▪▪▪ [2/3] Lowering...Done in 7.16337s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 0.31032142s
Done in 7.4865813s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.001077142s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.02613672s
[5/6] 🔍   Compiling from gir to lir
Done in 0.012959026s
[6/6] 🔍   Compiling from lir to enf
Done in 0.058442567s
✨  Finished in 7.642151s
.

======================================================= 10 passed in 76.17s (0:01:16) ========================================================

```

# Installing
**Requirements**

* Python >= 3.8

Download the latest release from https://github.com/furiosa-ai/furiosa-server/releases.
```
pip install furiosa_server-x.y.z-cp38-cp38-linux_x86_64.whl
```

## Usages

### Command lines
`furiosa-server` command has the following options.
To print out the command line usage, you can run `furiosa-server --help` option.
```sh
Usage: furiosa-server [OPTIONS]

  Start serving models from FuriosaAI model server

Options
  --log-level                 [ERROR|INFO|WARN|DEBUG|TRACE]    [default: LogLevel.INFO]
  --model-name                TEXT                             Model name [default: None]
  --model-path                TEXT                             Path to a model file (tflite, onnx are supported)
                                                               [default: None]
  --model-version             TEXT                             Model version [default: default]
  --host                      TEXT                             IPv4 address to bind [default: 0.0.0.0]
  --http-port                 INTEGER                          HTTP port to bind [default: 8080]
  --model-config              FILENAME                         Path to a model config file [default: None]
  --server-config             FILENAME                         Path to a server config file [default: None]
  --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]
  --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or
                                                               customize the installation.
                                                               [default: None]
  --help                                                       Show this message and exit.
```

### Serving a single model
To serve a single model, you will need only a couple of command line options.
The following is an example to start a model server with the specific model name and the model image file:

```sh
$ furiosa-server --model-name mnist --model-path samples/data/MNIST_inception_v3_quant.tflite --model-version 1
find native library /home/ys/Furiosa/compiler/npu-tools/target/x86_64-unknown-linux-gnu/release/
INFO:furiosa.runtime._api.v1:loaded dynamic library /home/ys/Furiosa/compiler/npu-tools/target/x86_64-unknown-linux-gnu/release/libnux.so (0.4.0-dev bdde0748b)
[1/6] 🔍   Compiling from tflite to dfg
Done in 0.04330982s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 38.590836s
▪▪▪▪▪ [2/3] Lowering...Done in 26.293291s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 2.2485964s
Done in 67.13952s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.000349475s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.07628228s
[5/6] 🔍   Compiling from gir to lir
Done in 0.002296112s
[6/6] 🔍   Compiling from lir to enf
Done in 0.06429358s
✨  Finished in 67.361084s
INFO:     Started server process [235857]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

You can find and try APIs via openapi: http://localhost:8080/docs#/

### Serving multiple models
To serve multiple models, you need to write a model configuration file.
The following is an example file located at `samples/model_config_example.yml`:
```yml
model_config_list:
  - name: mnist
    path: "samples/data/MNISTnet_uint8_quant.tflite"
    version: 1
    npu_device: npu0pe0
    compiler_config:
      keep_unsignedness: true
      split_unit: 0
  - name: ssd
    path: "samples/data/tflite/SSD512_MOBILENET_V2_BDD_int_without_reshape.tflite"
    version: 1
    npu_device: npu1
```

In a model configuration file, you can also specify a NPU device name dedicated to serve a certain model,
and a list of compiler configs as shown in the above example.

If you write a model config file,
you can launch the model server with a specific model config file as follow:
```sh
$ furiosa-server --model-config samples/model_config_example.yaml
find native library /home/ys/Furiosa/compiler/npu-tools/target/x86_64-unknown-linux-gnu/release/
INFO:furiosa.runtime._api.v1:loaded dynamic library /home/ys/Furiosa/compiler/npu-tools/target/x86_64-unknown-linux-gnu/release/libnux.so (0.4.0-dev bdde0748b)
[1/6] 🔍   Compiling from tflite to dfg
Done in 0.000510351s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 1.5242418s
▪▪▪▪▪ [2/3] Lowering...Done in 0.41843188s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 0.00754911s
Done in 1.9507353s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.000069757s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.005654631s
[5/6] 🔍   Compiling from gir to lir
Done in 0.000294499s
[6/6] 🔍   Compiling from lir to enf
Done in 0.003239762s
✨  Finished in 1.9631383s
[1/6] 🔍   Compiling from tflite to dfg
Done in 0.010595854s
[2/6] 🔍   Compiling from dfg to ldfg
▪▪▪▪▪ [1/3] Splitting graph...Done in 36.860104s
▪▪▪▪▪ [2/3] Lowering...Done in 8.500944s
▪▪▪▪▪ [3/3] Precalculating operators...Done in 1.2011535s
Done in 46.564877s
[3/6] 🔍   Compiling from ldfg to cdfg
Done in 0.000303809s
[4/6] 🔍   Compiling from cdfg to gir
Done in 0.07403221s
[5/6] 🔍   Compiling from gir to lir
Done in 0.001839668s
[6/6] 🔍   Compiling from lir to enf
Done in 0.07413657s
✨  Finished in 46.771423s
INFO:     Started server process [245257]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Submitting inference tasks

The following is an example of a request message. If you want to know the schema of the request message,
please refer to openapi specication.
```
{"inputs": [{"name": "mnist", "datatype": "INT32", "shape": [1, 1, 28, 28], "data": ...}]}

```

You can test one of MNIST model with the following command:
```
$ curl -X POST -H "Content-Type: application/json" \
-d "@samples/mnist_input_sample_01.json" \
http://localhost:8080/v2/models/mnist/versions/1/infer

{"model_name":"mnist","model_version":"1","id":null,"parameters":null,"outputs":[{"name":"0","shape":[1,10],"datatype":"UINT8","parameters":null,"data":[0,0,0,1,0,255,0,0,0,0]}]}%
```

Also, you can run a simple Python code to request the prediction task to the furiosa-server. Here is an example:
```python
import requests
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
url = 'http://localhost:8080/v2/models/mnist/versions/1/infer'
data = np.ndarray(x_train[0:1], dtype=np.uint8).flatten().tolist()
tensor = {
        'dataType': 'INT32',
        'shape': [1,1,28,28],
        'data': data
}
request = {'inputs': [tensor] }
response = requests.post(url, json=request)
print(response.json())
```
