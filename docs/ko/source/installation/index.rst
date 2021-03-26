**********************************
FuriosaAI SDK Installation
**********************************

.. note::

  FuriosaAI SDK의 일부 소프트웨어 (커널 드라이버, 펌웨어, 런타임)는
  FuriosaAI SDK 및 NPU 평가 프로그램을 통해 별도의 채널로 전달 받아야 한다.
  FuriosaAI SDK 및 NPU 평가 프로그램 등록은 contact@furiosa.ai 로 신청할 수 있다.


SDK 설치를 위한 최소 요구사항

  * Ubuntu 18.04 LTS (Bionic Beaver) 또는 Debian buster
    또는 상위 버전
  * 시스템의 관리자 권한 (root)
  * Python 3.7+ 및 최신 버전의 PIP

    .. code-block::

      $ pip3 install --upgrade pip

  * GitHub 및 PyPi 로 연결 가능한 네트워크 환경
  * build-essential 및 cmake 설치

    .. code-block::

      $ apt-get update
      $ apt-get install cmake build-essential

  * onnxruntime 1.6.0 (모델 quantization & calibration, NPU 에서 지원하지 않는 오퍼레이터 처리를 위해 설치 필요)

    .. code-block::

      $ wget https://github.com/hyunsik/onnxruntime/releases/download/v1.6.0/libonnxruntime-1.6.0_amd64.deb
      $ apt-get install -y ./libonnxruntime-1.6.0_amd64.deb


SDK 구성 요소별 설치 안내

  * :doc:`FuriosaAI NPU 커널 드라이버 및 펌웨어 설치 [Required]<driver>` : NPU 를 구동하기 위한 커널 드라이버 및 펌웨어

  * :doc:`FuriosaAI NPU Runtime 설치 [Required]<runtime>` : NPU 런타임 및 NPU 응용 프로그램 도구

  * :doc:`FuriosaAI NPU Python SDK 설치 [Required]<python-sdk>` : NPU 사용을 위한 Python 라이브러리 및 명령도구 (cli)

    * :doc:`FuriosaAI NPU Python SDK 설치를 위한 Python 환경 구성 [Optional]<python-sdk-setup-python>` : FuriosaAI NPU Python SDK 설치를 위한 Python 3.7+ 환경 구성 안내

  * :doc:`FuriosaAI API 키 설정 [Optional]<apikey>` : FuriosaAI에서 웹 서비스로 제공하는 도구를 사용하기 위한 설정



.. toctree::
  :hidden:

  driver
  runtime
  python-sdk
  apikey
