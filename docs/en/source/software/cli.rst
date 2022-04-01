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

  .. tab:: 다운로드 센터를 이용한 설치

    아래 패키지들의 최신 버전을 선택하여 다운 받아 명령에 쓰여진 순서대로 설치한다.

    * furiosactl

    .. code-block:: sh

      sudo apt-get install -y ./furiosa-toolkit-x.y.z-?.deb


furiosactl 사용법
----------------------------------------

커널 드라이버 설치 후 NPU 장치가 잘 인식되었는지 확인하기 위해 ``furiosactl`` 명령을 사용할 수 있다.
현재 이 명령은 NPU 장치의 DEVICE ID와 온도, PCI 정보 출력하는 ``furiosactl info`` 명령을 제공하고 있다.


.. code-block:: sh

  furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu1 | FuriosaAI Warboy |  40°C | 0.00 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+


furiosa
===================================

``furiosa`` 커맨드는 `Python SDK <PythonSDK>` 를 설치하면 사용할 수 있는 메타 명령형 도구이다.
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

``litmus`` 명령은 `TFLite`_, `ONNX`_ 모델을 인자로 받아,
자동으로 양자화한 후 최종 바이너리까지 컴파일을 시도하여 주어진 모델이 SDK와 호환되는지 검사한다.

.. code-block:: sh

  $ furiosa litmus yolov4.onxx
  [Step 1] Checking if the model can be transformed into a quantized model ...
  Quantization: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 85.33it/s]
  [Step 1] Passed
  [Step 2] Checking if the model can be compiled to a NPU program ...
  [Step 2] Passed


실패하는 경우 아래와 같은 오류를 볼 수 있으며 오류가 발생한 경우 메시지를
`FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_ 에
`버그 신고(Bug Report)` 섹션 보고하여 지원을 받을 수 있다.

.. code-block:: sh

  $ furiosa litmus efficientnet-lite4-11.onnx

    Stdout:
    [Step 1] Checking if the model can be transformed into a quantized model ...

    Stderr:
    /root/miniconda3/envs/furiosa/lib/python3.8/site-packages/onnx/__init__.py:97: RuntimeWarning: Unexpected end-group tag: Not all data was converted
        decoded = cast(Optional[int], proto.ParseFromString(s))
    [Step 1] Failed
