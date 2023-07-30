*********************************************************
명령행 도구
*********************************************************

FuriosaAI SDK는 NPU 디바이스 정보를 출력 하거나 모델 컴파일, 모델과 SDK 간의 호환성 확인
등의 기능을 명령행 도구로 제공한다. 이 섹션에서는 각 명령형 도구 별 설치 방법과 사용 방법에 대해 설명한다.

.. _Toolkit:

furiosa-toolkit
===================================
``furiosa-toolkit`` 은 NPU 장치를 관리하고 정보를 확인하는 명령형 도구를 제공한다.


furiosa-toolkit 설치
--------------------------------------
이 명령형 도구 사용을 위해서는 사전에 :ref:`RequiredPackages` 를 따라 커널 드라이버를 설치해야 한다.
그 이후에는 아래 설명을 따라 furiosa-toolkit 을 설치한다.

.. tabs::

  .. tab:: APT 서버를 이용한 설치

    .. code-block:: sh

      sudo apt-get install -y furiosa-toolkit



furiosactl 사용법
----------------------------------------
furiosactl 커맨드는 다양한 서브 커맨드를 제공하고 장치의 정보를 얻거나 제어하는 기능을 가지고 있다.

문법 개요:

.. code-block:: sh

    furiosactl <sub command> [option] ..

``furiosactl info``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``info`` 서브 커맨드를 통해 머신이 인식한 NPU 장치의 정보를 확인할 수 있다.
이 명령은 NPU 장치의 온도, PCI 정보 등을 출력한다. 만약 머신에 장치를 장착한 후에도 장치가 이 명령으로 보이지 않는다면,
:ref:`RequiredPackages` 를 따라 드라이버 설치해야 한다.
``info`` 커맨드와 함께 ``--full`` 옵션을 추가하면 장치의 UUID, Serial Number 정보를 함께 확인할 수 있다.

.. code-block:: sh

  $ furiosactl info
  +------+--------+----------------+-------+--------+--------------+
  | NPU  | Name   | Firmware       | Temp. | Power  | PCI-BDF      |
  +------+--------+----------------+-------+--------+--------------+
  | npu1 | warboy | 1.6.0, 3c10fd3 |  54°C | 0.99 W | 0000:44:00.0 |
  +------+--------+----------------+-------+--------+--------------+

  $ furiosactl info --full
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+
  | NPU  | Name   | UUID                                 | S/N               | Firmware       | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+
  | npu1 | warboy | 00000000-0000-0000-0000-000000000000 | WBYB0000000000000 | 1.6.0, 3c10fd3 |  54°C | 0.99 W | 0000:44:00.0 | 511:0   |
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+

``furiosactl list``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``list`` 서브 커맨드는 NPU 장치에서 사용할 수 있는 device file의 정보를 제공한다.
NPU에 존재하는 각 코어가 사용 중인지 유휴 상태인지 여부를 확인할 수도 있다.

.. code-block:: sh

  furiosactl list
  +------+------------------------------+-----------------------------------+
  | NPU  | Cores                        | DEVFILES                          |
  +------+------------------------------+-----------------------------------+
  | npu1 | 0 (available), 1 (available) | npu1, npu1pe0, npu1pe1, npu1pe0-1 |
  +------+------------------------------+-----------------------------------+

``furiosactl ps``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``ps`` 서브 커맨드는 현재 NPU 장치를 점유하고 있는 OS 프로세스의 정보를 출력해준다.

.. code-block:: sh

    $ furiosactl ps
    +-----------+--------+------------------------------------------------------------+
    | NPU       | PID    | CMD                                                        |
    +-----------+--------+------------------------------------------------------------+
    | npu0pe0-1 | 132529 | /usr/bin/python3 /usr/local/bin/uvicorn image_classify:app |
    +-----------+--------+------------------------------------------------------------+


``furiosactl top`` (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``top`` 서브 커맨드는 시간의 흐름에 따른 NPU 장치 별 사용률을 확인하는데 사용한다.
출력 항목은 다음의 의미를 나타낸다.
기본적으로 1초 간격으로 사용률을 계산하지만, ``--interval`` 옵션을 통해 계산 주기를 직접 설정할 수 있다. (단위: ms)

.. list-table:: furiosa top fields
   :widths: 100 400
   :header-rows: 1

   * - 항목
     - 설명
   * - Datetime
     - 관측 시각
   * - PID
     - NPU를 사용 중인 프로세스ID
   * - Device
     - 사용 중인 NPU 장치
   * - NPU(%)
     - 관측 시간동안 NPU가 사용된 시간의 비율
   * - Comp(%)
     - NPU가 사용된 시간 중 연산에 사용된 시간의 비율
   * - I/O(%)
     - NPU가 사용된 시간 중 I/O에 사용된 시간의 비율
   * - Command
     - 프로세스의 실행 명령행


.. code-block:: sh

    $ furiosactl top --interval 200
    NOTE: furiosa top is under development. Usage and output formats may change.
    Please enter Ctrl+C to stop.
    Datetime                        PID       Device        NPU(%)   Comp(%)   I/O(%)   Command
    2023-03-21T09:45:56.699483936Z  152616    npu1pe0-1      19.06    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:56.906443888Z  152616    npu1pe0-1      51.09     93.05     6.95   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.110489333Z  152616    npu1pe0-1      46.40     97.98     2.02   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.316060982Z  152616    npu1pe0-1      51.43    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.521140588Z  152616    npu1pe0-1      54.28     94.10     5.90   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.725910558Z  152616    npu1pe0-1      48.93     98.93     1.07   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.935041998Z  152616    npu1pe0-1      47.91    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:58.13929122Z   152616    npu1pe0-1      49.06     94.94     5.06   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf


furiosa
===================================

``furiosa`` 커맨드는 :ref:`Python SDK <PythonSDK>` 를 설치하면 사용할 수 있는 메타 명령형 도구이다.
또한 확장 패키지를 설치하면 추가 하위 커맨드(subcommand) 가 추가 된다.

만약 Python 실행 환경이 준비되어 있지 않다면 :any:`SetupPython` 를 참고한다.


명령행 도구 설치

.. code-block:: sh

  $ pip install furiosa-sdk


설치 확인

.. code-block:: sh

  $ furiosa compile --version
  libnpu.so --- v2.0, built @ fe1fca3
  0.5.0 (rev: 49b97492a built at 2021-12-07 04:07:08) (wrapper: None)


furiosa compile
--------------------

``compile`` 명령은 `TFLite <https://www.tensorflow.org/lite>`_, `ONNX <https://onnx.ai/>`_
형식의 모델을 컴파일하여 FuriosaAI NPU를 사용하는 프로그램을 생성한다.
자세한 설명과 옵션은 :ref:`CompilerCli` 페이지에서 찾을 수 있다.

.. _Litmus:

furiosa litmus (모델 적합 여부 검사)
--------------------------------------------

``litmus`` 명령은 `ONNX`_ 모델을 받아 Furiosa SDK 및 Furiosa NPU와 호환되는지 빠르게 검사할 수 있는 도구이다.
``litmus``는 원본 ONNX 모델로부터 SDK를 이용해 추론하는 전 과정을 수행하고 각 과정이 잘 동작하는지 확인한다. ``litmus`` 는 버그 리포팅에도 유용하게 쓸 수 있다.
``--dump`` 옵션을 주면, 각 과정에서 생성되는 로그와 환경 정보를 수집하여 zip 파일로 묶어 출력한다. 버그 리포팅 시에 이 파일을 같이 첨부하면 도움을 더 빠르게 받을 수 있다.

``litmus`` 명령이 실행하는 단계는 다음과 같다.

  - Step1: 입력받은 ONNX 모델을 불러오고 유효한 모델인지 검사한다.
  - Step2: 임의의 데이터를 보정 범위(calibration range)로 적용하여 양자화를 수행한다.
  - Step3: 양자화된 모델을 NPU에서 가속할 수 있도록 컴파일을 시도한다.
  - Step4: 컴파일된 모델을 ``furiosa-bench`` 를 이용해 실행시킨다. 만약 ``furiosa-bench`` 가 없다면 이 단계는 생략된다.


문법 개요:

.. code-block:: sh

  furiosa-litmus [-h] [--dump OUTPUT_PREFIX] [--skip-quantization] [--target-npu TARGET_NPU] [-v] model_path

``litmus`` 명령의 간단한 사용 예시는 다음과 같다.

.. code-block:: sh

  $ furiosa litmus model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Checking if the model can be loaded and optimized ...
  [Step 1] Passed
  [Step 2] Checking if the model can be quantized ...
  [Step 2] Passed
  [Step 3] Checking if the model can be compiled for the NPU family [warboy-2pe] ...
  [1/6] 🔍   Compiling from onnx to dfg
  Done in 0.09272794s
  [2/6] 🔍   Compiling from dfg to ldfg
  ▪▪▪▪▪ [1/3] Splitting graph(LAS)...Done in 9.034934s
  ▪▪▪▪▪ [2/3] Lowering graph(LAS)...Done in 20.140083s
  ▪▪▪▪▪ [3/3] Optimizing graph...Done in 0.019548794s
  Done in 29.196825s
  [3/6] 🔍   Compiling from ldfg to cdfg
  Done in 0.001701888s
  [4/6] 🔍   Compiling from cdfg to gir
  Done in 0.015205072s
  [5/6] 🔍   Compiling from gir to lir
  Done in 0.0038304s
  [6/6] 🔍   Compiling from lir to enf
  Done in 0.020943863s
  ✨  Finished in 29.331545s
  [Step 3] Passed
  [Step 4] Perform inference once for data collection... (Optional)
  ✨  Finished in 0.000001198s
  ======================================================================
  This benchmark was executed with latency-workload which prioritizes latency of individual queries over throughput.
  1 queries executed with batch size 1
  Latency stats are as follows
  QPS(Throughput): 125.00/s

  Per-query latency:
  Min latency (us)    : 7448
  Max latency (us)    : 7448
  Mean latency (us)   : 7448
  50th percentile (us): 7448
  95th percentile (us): 7448
  99th percentile (us): 7448
  99th percentile (us): 7448
  [Step 4] Finished


이미 양자화된 모델을 가지고 있다면 ``--skip-quantization`` 옵션을 사용하여 양자화 과정을 생략할 수 있다.


.. code-block:: sh

  $ furiosa litmus --skip-quantization quantized-model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Skip model loading and optimization
  [Step 2] Skip model quantization
  [Step 1 & Step 2] Load quantized model ...
  [Step 3] Checking if the model can be compiled for the NPU family [warboy-2pe] ...
  ...


``--dump <path>`` 옵션을 사용하여 컴파일 로그, 런타임 로그, 소프트웨어 버전 및 실행환경 등 분석에 필요한 메타데이터를 모은 `<path>-<unix_epoch>.zip` 파일을 생성할 수 있다.
컴파일에 실패하거나 실행에 실패하는 등 문제가 있다면 이 파일을 가지고 `FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portal/1>`_ 을 통해 지원을 받을 수 있다.


.. code-block:: sh

  $ furiosa litmus --dump archive model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Checking if the model can be loaded and optimized ...
  [Step 1] Passed
  ...

  $ zipinfo -1 archive-1690438803.zip 
  archive-16904388032l4hoi3h/meta.yaml
  archive-16904388032l4hoi3h/compiler/compiler.log
  archive-16904388032l4hoi3h/compiler/memory-analysis.html
  archive-16904388032l4hoi3h/compiler/model.dot
  archive-16904388032l4hoi3h/runtime/trace.json