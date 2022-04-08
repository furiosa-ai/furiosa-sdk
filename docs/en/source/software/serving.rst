.. _ModelServing:

**********************************************************
Model Server (Serving Framework)
**********************************************************

To serve DNN models through GRPC and REST API, you can use `Furiosa Model Server <https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server>`_.
Model Server provides the endpoints compatible with `KServe Predict Protocol Version 2 <https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md>`_.

Its major features are:

 * REST/GRPC endpoints support
 * Multiple model serving using multiple NPU devices


Installation
============================

Its requirements are:

* Ubuntu 18.04 LTS (Debian buster) or higher
* :ref:`RequiredPackages`
* Python 3.7 or higher version

If you need Python environment, please refer to :ref:`SetupPython` first.


.. tabs::

   .. tab:: Installation using PIP

      Run the following command

      .. code-block:: sh

        $ pip install 'furiosa-sdk[server]'

   .. tab:: Installation from source code

      Check out the source code and run the following command

      .. code-block:: sh

        $ git clone https://github.com/furiosa-ai/furiosa-sdk.git
        $ cd furiosa-sdk/python/furiosa-server
        $ pip install .



Running a Model Server
============================

You can run model sever command by running ``furiosa server`` in your shell.


To run simply a model server with ``tflite`` or ``onnx``, you need to specify
just the model path and its name as following:

.. code-block:: sh

    $ cd furiosa-sdk
    $ furiosa server \
    --model-path examples/assets/quantized_models/MNISTnet_uint8_quant_without_softmax.tflite \
    --model-name mnist


``--model-path`` option allows to specify a path of a model file.
If you want to use a specific binding address and port, you can use additionally
``--host``, ``--host-port``.

Please run ``furiosa server --help`` if you want to learn more
about the command with various options.


.. code-block:: sh

    $ furiosa server --help
    Usage: furiosa server [OPTIONS]

        Start serving models from FuriosaAI model server

    Options:
        --log-level [ERROR|INFO|WARN|DEBUG|TRACE]
                                        [default: LogLevel.INFO]
        --model-path TEXT               Path to Model file (tflite, onnx are
                                        supported)
        --model-name TEXT               Model name used in URL path
        --model-version TEXT            Model version used in URL path  [default:
                                        default]
        --host TEXT                     IP address to bind  [default: 0.0.0.0]
        --http-port INTEGER             HTTP port to listen to requests  [default:
                                        8080]
        --model-config FILENAME         Path to a config file about models with
                                        specific configurations
        --server-config FILENAME        Path to Model file (tflite, onnx are
                                        supported)
        --install-completion [bash|zsh|fish|powershell|pwsh]
                                        Install completion for the specified shell.
        --show-completion [bash|zsh|fish|powershell|pwsh]
                                        Show completion for the specified shell, to
                                        copy it or customize the installation.
        --help                          Show this message and exit.


Running a Model Server with a Configuration File
=============================================================

If you need more advanced configurations like compilation options and device options,
you can use a configuration file based on Yaml.


.. code-block:: yaml

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
        npu_device: npu0pe1

When you run a model sever with a configuration file,
you need to specify ``--model-config`` as following.
You can find the model files described in the above example from
`furiosa-models/samples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server/samples>`_.

.. code-block:: sh

    $ cd furiosa-sdk/python/furiosa-server
    $ furiosa server --model-config samples/model_config_example.yaml

    Saving the compilation log into /Users/hyunsik/.local/state/furiosa/logs/compile-20211126143917-2731kz.log
    Using furiosa-compiler 0.5.0 (rev: 407c0c51f built at 2021-11-26 12:05:30)
    2021-11-26T22:39:17.819518Z  INFO Npu (npu0pe0) is being initialized
    2021-11-26T22:39:17.823511Z  INFO NuxInner create with pes: [PeId(0)]
    ...
    INFO:     Started server process [62087]
    INFO:uvicorn.error:Started server process [62087]
    INFO:     Waiting for application startup.
    INFO:uvicorn.error:Waiting for application startup.
    INFO:     Application startup complete.
    INFO:uvicorn.error:Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
    INFO:uvicorn.error:Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

Once a model server starts up, you can call the inference request through HTTP protocol.
If the model name is ``mnist`` and its version ``1``, the endpoint of the model will be
``http://<host>:<port>/v2/models/mnist/version/1/infer``, accepting ``POST`` http request.
The following is an example using ``curl`` to send the inference request and return the response.

.. code-block: sh

    $ curl -X POST -H "Content-Type: application/json" \
    -d "@samples/mnist_input_sample_01.json" \
    http://localhost:8080/v2/models/mnist/versions/1/infer

    {"model_name":"mnist","model_version":"1","id":null,"parameters":null,"outputs":[{"name":"0","shape":[1,10],"datatype":"UINT8","parameters":null,"data":[0,0,0,1,0,255,0,0,0,0]}]}


The following is a Python example, doing same as ``curl`` does in the above example.

.. code-block:: python

    import requests
    import mnist
    import numpy as np

    mnist_images = mnist.train_images().reshape((60000, 1, 28, 28)).astype(np.uint8)
    url = 'http://localhost:8080/v2/models/mnist/versions/1/infer'

    data = mnist_images[0:1].flatten().tolist()
    request = {
        "inputs": [{
            "name":
            "mnist",
            "datatype": "UINT8",
            "shape": (1, 1, 28, 28),
            "data": data
        }]
    }

    response = requests.post(url, json=request)
    print(response.json())


Endpoints
=======================================
The following table shows REST API endpoints and its descriptions.
The model server is following KServe Predict Protocol Version 2.
So, you can find more details from `KServe Predict Protocol Version 2 - HTTP/REST <https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#httprest>`_.

.. list-table:: Endpoints of KServe Predict Protocol Version 2
   :widths: 50 50
   :header-rows: 1

   * - Method and Endpoint
     - Description
   * - GET /v2/health/live
     - Returns HTTP Ok (200) if the inference server is able to receive and respond to metadata and inference requests.
       This API can be directly used for the Kubernetes livenessProbe.
   * - GET /v2/health/ready
     - Returns HTTP Ok (200) if all the models are ready for inferencing.
       This API can be directly used for the Kubernetes readinessProbe.
   * - GET /v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}
     - Returns a model metadata
   * - GET /v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}/ready
     - Returns HTTP Ok (200) if a specific model is ready for inferencing.
   * - POST /v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer
     - Inference request