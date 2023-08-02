.. _ModelServing:

**********************************************************
모델 서버 (서빙 프레임워크)
**********************************************************

준비된 모델을 실제 서비스 환경에 배포할 때 모델을 GRPC나 REST API를 통하는 경우가 일반적이다.
이런 유스케이스를 위해 FuriosaAI SDK는 `KServe Predict Protocol Version 2 <https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md>`_ 를 지원하는
모델 서버를 제공한다.

모델 서버는 다음 주요 기능을 제공한다.

 * REST API 지원
 * 다수의 NPU 디바이스 및 다수의 모델를 하나의 서버로 서빙 지원


모델 서버 설치
============================

모델 서버 설치를 위한 최소 요구사항은 다음과 같다.

* Ubuntu 20.04 LTS (Debian bullseye) 또는 상위 버전
* :ref:`FuriosaAI SDK 필수 패키지 <RequiredPackages>`
* Python 3.8 또는 상위 버전

Python 실행환경 준비가 필요하다면 :ref:`SetupPython` 를 참고한다.


.. tabs::

   .. tab:: PIP를 이용한 설치

      간단하게 다음 커맨드를 실행해주세요.

      .. code-block:: sh

        $ pip install 'furiosa-sdk[server]'

   .. tab:: 소스코드를 이용한 설치

      아래와 같이 Github에서 소스를 다운받아 설치한다.

      .. code-block:: sh

        $ git clone https://github.com/furiosa-ai/furiosa-sdk.git
        $ cd furiosa-sdk/python/furiosa-server
        $ pip install .



모델 서버 실행
============================

모델 서버는 명령행 도구 ``furiosa server`` 커맨드를 통해 실행 할 수 있다.
``furiosa server --help`` 을 실행하면 아래와 같은 도움말을 볼 수 있다.

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


간단한 모델 서빙은 커맨드로 ``tflite``, ``onnx`` 포맷의 모델 이미지의 패스와 모델 이름을
지정하면 실행할 수 있다.

.. code-block:: sh

    $ cd furiosa-sdk
    $ furiosa server \
    --model-path examples/assets/quantized_models/MNISTnet_uint8_quant_without_softmax.tflite \
    --model-name mnist


``--model-path`` 옵션으로 로컬 파일 시스템에 저장된 모델을 지정할 수 있다.
또한 모델 서버가 지정한 호스트 이름과 포트로 연결 요청을 대기하기 원하는
경우 ``--host``, ``--host-port`` 로 각각 설정할 수 있다.


모델 설정을 이용한 모델 서버 실행
=================================

컴파일 옵션이나 서빙에 대한 더 고급 설정이 필요한 경우 또는 반복적으로 같은 옵션을 사용하는 경우
모델 설정을 활용할 수 있다.


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


위와 같은 설정을 준비한 뒤에 아래와 같이 ``--model-config`` 옵션을 이용하여
설정 파일의 패스를 지정하여 실행할 수 있다. 위 예제 실행을 위해서는 모델이 필요한데 위 모델은
`Furiosa Server Github 저장소 <https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server>`_
의 `samples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/python/furiosa-server/samples>`_ 디렉토리에서
찾을 수 있다. 모델과 모델 설정이 준비되어 있다면 아래 예제를 실행해볼 수 있다.

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


모델 서버가 시작되고 나면 일반적인 HTTP 요청을 보내 모델의 추론 작업을 해볼 수 있다.
모델 설정에서 모델 이름이 ``mnist`` 이고 버전이 ``1`` 인 모델에 추론 요청을 보낼 때는
``http://<host>:<port>/v2/models/mnist/version/1/infer`` 에 ``POST`` 요청을 보내면 된다.

.. code-block: sh

    $ curl -X POST -H "Content-Type: application/json" \
    -d "@samples/mnist_input_sample_01.json" \
    http://localhost:8080/v2/models/mnist/versions/1/infer

    {"model_name":"mnist","model_version":"1","id":null,"parameters":null,"outputs":[{"name":"0","shape":[1,10],"datatype":"UINT8","parameters":null,"data":[0,0,0,1,0,255,0,0,0,0]}]}


아래 예제는 위와 동일한 요청을 Python 코드를 통해 보내는 예제이다.

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


엔드포인트(Endpoint) 정보
=======================================
다음 테이블은 모델 서버가 제공하는 주요 REST API 엔드포인트 정보이다.
모델 서버는 `KServe Predict Protocol Version 2 - HTTP/REST <https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#httprest>`_
를 따르고 있으므로 더 자세한 정보는 링크에서 찾아볼 수 있다.

.. list-table:: Endpoints of KServe Predict Protocol Version 2
   :widths: 50 50
   :header-rows: 1

   * - Method and Endpoint
     - Description
   * - GET /v2/health/live
     - 서버가 요청을 처리할 수 있는 상태면 HTTP 상태 Ok 리턴 (Kubernetes livenessProbe에 해당)
   * - GET /v2/health/ready
     - 모든 모델이 추론 작업을 위한 준비가 되면 HTTP 상태 Ok 리턴 (Kubernetes readinessProbe에 해당)
   * - GET /v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}
     - 모델 메타데이터 반환
   * - GET /v2/models/${MODEL_NAME}/versions/${MODEL_VERSION}/ready
     - 특정 버전의 모델이 추론 요청을 처리할 준비가 되었다면 HTTP 상태 Ok 리턴
   * - POST /v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer
     - 추론 요청
