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

* Ubuntu 20.04 LTS (Debian bullseye) or higher
* :ref:`RequiredPackages`
* Python 3.8 or higher version

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
        model: "samples/data/MNISTnet_uint8_quant.tflite"
        version: "1"
        platform: npu
        npu_device: warboy(1)*1
        compiler_config:
          keep_unsignedness: true
          split_unit: 0
      - name: ssd
        model: "samples/data/SSD512_MOBILENET_V2_BDD_int_without_reshape.tflite"
        version: "1"
        platform: npu
        npu_device: warboy(1)*1

When you run a model sever with a configuration file,
you need to specify ``--model-config`` as following.
You can find the model files described in the above example from
`furiosa-models/samples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server/samples>`_.

.. code-block:: sh

    $ cd furiosa-sdk/python/furiosa-server
    $ furiosa server --model-config samples/model_config_example.yaml
    libfuriosa_hal.so --- v0.11.0, built @ 43c901f
    2023-08-02T07:42:42.263133Z  INFO furiosa_rt_core::driver::event_driven::device: DeviceManager has detected 1 NPUs
    2023-08-02T07:42:42.267247Z  INFO furiosa_rt_core::driver::event_driven::device: [1] npu:6:1 (warboy-b0, 64dpes)
    2023-08-02T07:42:42.267264Z  INFO furiosa_rt_core::driver::event_driven::coord: furiosa-rt (v0.10.0-rc6, rev: d021ff71d, built_at: 2023-07-31T19:05:26Z) is being initialized
    2023-08-02T07:42:42.267269Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:1-io-thread-0 thread has started
    2023-08-02T07:42:42.267398Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:1-commit-thread thread has started
    2023-08-02T07:42:42.267405Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:1-io-thread-1 thread has started
    2023-08-02T07:42:42.270837Z  INFO furiosa_rt_core::driver::event_driven::coord: Loaded libcompiler 0.10.0 (rev: f8f05c built: 2023-07-26T09:49:17Z)
    2023-08-02T07:42:42.270851Z  INFO furiosa_rt_core::driver::event_driven::coord: Loaded libhal-warboy 0.11.0 (rev: 43c901f built: 2023-04-19T14:04:55Z)
    2023-08-02T07:42:42.271144Z  INFO furiosa_rt_core::driver::event_driven::coord: [NONAME] Runtime has started
    2023-08-02T07:42:42.273772Z  INFO furiosa_rt_core::driver::event_driven::coord: Model#0001 is being loaded to npu:6:1
    2023-08-02T07:42:42.283260Z  INFO furiosa_rt_core::driver::event_driven::coord: Compiling Model#0001 (target: warboy-b0, 64dpes, file: MNISTnet_uint8_quant.tflite, size: 18.2 kiB)
    2023-08-02T07:42:42.299091Z  INFO furiosa_rt_core::driver::event_driven::coord: Model#0001 has been compiled successfully (took 0 secs)
    2023-08-02T07:42:42.299293Z  INFO furiosa_rt_core::dag: Task Statistics: TaskStats { cpu: 5, npu: 1, alias: 0, coalesce: 0 }
    2023-08-02T07:42:42.300701Z  INFO furiosa_rt_core::driver::event_driven::coord: NpuApi (AsyncNpuApiImpl) has started..
    2023-08-02T07:42:42.300721Z  INFO furiosa_rt_core::driver::event_driven::coord: Creating 1 Contexts on npu:6:1 (DRAM usage: 6.0 kiB / 16.0 GiB, SRAM usage: 124.0 kiB / 64.0 MiB)
    2023-08-02T07:42:42.300789Z  INFO furiosa_rt_core::driver::event_driven::coord: npu:6:1 has scheduled to Model#0001
    2023-08-02T07:42:42.304216Z  WARN furiosa_rt_core::consts::envs: NPU_DEVNAME will be deprecated. Use FURIOSA_DEVICES instead.
    2023-08-02T07:42:42.313084Z  INFO furiosa_rt_core::driver::event_driven::device: DeviceManager has detected 1 NPUs
    2023-08-02T07:42:42.315470Z  INFO furiosa_rt_core::driver::event_driven::device: [1] npu:6:0 (warboy-b0, 64dpes)
    2023-08-02T07:42:42.315483Z  INFO furiosa_rt_core::driver::event_driven::coord: furiosa-rt (v0.10.0-rc6, rev: d021ff71d, built_at: 2023-07-31T19:05:26Z) is being initialized
    2023-08-02T07:42:42.315560Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:0-io-thread-1 thread has started
    2023-08-02T07:42:42.315610Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:0-io-thread-0 thread has started
    2023-08-02T07:42:42.315657Z  INFO furiosa_rt_core::npu::async_impl::threaded: npu:6:0-commit-thread thread has started
    2023-08-02T07:42:42.319127Z  INFO furiosa_rt_core::driver::event_driven::coord: Loaded libcompiler 0.10.0 (rev: f8f05c built: 2023-07-26T09:49:17Z)
    2023-08-02T07:42:42.319141Z  INFO furiosa_rt_core::driver::event_driven::coord: Loaded libhal-warboy 0.11.0 (rev: 43c901f built: 2023-04-19T14:04:55Z)
    2023-08-02T07:42:42.319364Z  INFO furiosa_rt_core::driver::event_driven::coord: [NONAME] Runtime has started
    2023-08-02T07:42:42.324283Z  INFO furiosa_rt_core::driver::event_driven::coord: Model#0002 is being loaded to npu:6:0
    2023-08-02T07:42:42.333521Z  INFO furiosa_rt_core::driver::event_driven::coord: Compiling Model#0002 (target: warboy-b0, 64dpes, file: SSD512_MOBILENET_V2_BDD_int_without_reshape.tflite, size: 5.2 MiB)
    2023-08-02T07:42:42.814260Z  INFO furiosa_rt_core::driver::event_driven::coord: Model#0002 has been compiled successfully (took 0 secs)
    2023-08-02T07:42:42.815406Z  INFO furiosa_rt_core::dag: Task Statistics: TaskStats { cpu: 26, npu: 1, alias: 0, coalesce: 0 }
    2023-08-02T07:42:42.893745Z  INFO furiosa_rt_core::driver::event_driven::coord: NpuApi (AsyncNpuApiImpl) has started..
    2023-08-02T07:42:42.893772Z  INFO furiosa_rt_core::driver::event_driven::coord: Creating 1 Contexts on npu:6:0 (DRAM usage: 1.0 MiB / 16.0 GiB, SRAM usage: 14.8 MiB / 64.0 MiB)
    2023-08-02T07:42:42.894265Z  INFO furiosa_rt_core::driver::event_driven::coord: npu:6:0 has scheduled to Model#0002
    INFO:     Started server process [2448540]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

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
