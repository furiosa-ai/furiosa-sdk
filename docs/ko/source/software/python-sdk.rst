.. _PythonSDK:

**********************************
Python SDK 설치 및 사용 가이드
**********************************

FuriosaAI Python SDK는 NPU를 사용하는 Python 응용을 작성하기 위한
소프트웨어 개발 키트이다. Python SDK를 이용하여 NPU 응용 애플리케이션 개발에
AI/ML 분야에서 가장 널리 사용되는 Python 생태계의 여러 도구와 라이브러리, 프레임워크들을 활용할 수 있다.
Python SDK는 다양한 모듈로 구성되며 추론 API, 양자화 API, 명령행 도구, 서빙을 위한 서버 프로그램을 제공한다.


설치 사전 요구 사항
=======================================================================
* Ubuntu 18.04 LTS (Debian buster) 또는 상위 버전
* :ref:`FuriosaAI SDK 필수 패키지 <RequiredPackages>`
* Python 3.7 또는 상위 버전 (Python 환경 구성은 :any:`SetupPython` 를 참고)
* pip 최신 버전

Python SDK 설치 및 사용을 위해서는 :ref:`필수 패키지 설치 <RequiredPackages>` 가이드를 따라
커널 드라이버, 펌웨어, 런타임 라이브러리를 반드시 설치해야 한다.


.. _SetupPython:

Python 실행 환경 구성
================================================================

Python SDK는 Python 3.7 또는 그 상위 버전의 실행 환경이 필요하고
이 섹션에서는 Python 실행 환경 구성을 설명한다.

.. note::

  FuriosaAI Python SDK를 사용하지 않거나 Python 실행 환경 구성에 익숙하다면 이 장을 건너뛰어도 좋다.


아래 명령으로 현재 시스템에 준비되어 있는 Python 버전을 확인할 수 있다.

.. code-block::

  python --version
  Python 3.8.10


만약 Python 명령이 존재하지 않거나 3.7 미만의 Python을 사용하고 있다면
아래 방법 중에 하나를 선택하여 Python 환경을 구성할 수 있다.

* :ref:`CondaInstall` (권장):
  `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ 는
  특정 Python 어플리케이션만을 위한 독립된 Python 환경을 구성할 수 있게 해준다.
  따라서 Conda를 사용하면 Python 어플리케이션을 설치할 때 종종 발생하는 패키지 의존성 문제나 Python 버전 문제를
  피할 수 있다.
* :ref:`SetupPythonOnLinux`: Linux 시스템에서 직접 Python 실행 환경을 구성한다.
  다른 Python 실행 환경과 충돌이 우려되지 않는 경우 선택할 수 있다.


.. _CondaInstall:

Conda를 이용한 Python 환경 구성
-------------------------------------------------------

Conda는 특정 Python 어플리케이션만을 위한 독립된 Python 환경을 쉽게 구성하게 해준다.
Conda에 대해 자세히 알고 싶다면 `Conda`_ 에서 다양한 문서를 참고할 수 있다.


설치는 아래와 같이 설치 프로그램을 다운 받아 시작할 수 있다.
``sh ./Miniconda3-latest-Linux-x86_64.sh`` 실행 시 물어보는 것은 모두 `yes` 를 선택하면 된다.

.. code-block::

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh ./Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  conda --version


독립된 Python 실행 환경 생성 및 활성화
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Anaconda 설치 후에는 독립된 Python 실행 환경을 구성하고 필요할 때 활성화 할 수 있다.

1. Python 3.8을 사용한다면 ``furiosa-3.8`` 라는 이름으로 아래 명령으로 실행 환경을 생성하자.

.. code-block::

  conda create -n furiosa-3.8 python=3.8


2. 생성한 Python 3.8 환경은 ``activate`` 명령으로 활성화된다.

.. code-block::

  conda activate furiosa-3.8
  # 버전 확인
  python --version


3. Python 실행 환경이 활성화 되고 나면 :ref:`InstallPipPackages` 를 따라 Python SDK를 설치 한다.


4. Python 실행 환경의 사용을 끝내고 싶은 경우 ``deactivate`` 명령을 사용한다.

.. code-block::

  $ conda deactivate

한번 생성한 환경은 언제든지 다시 ``activate`` 하여 사용할 수 있다.
이미 설치했던 패키지는 활성화 후에 다시 설치하지 않아도 된다.


.. _SetupPythonOnLinux:

Linux 패키지를 이용한 Python 환경 구성
-------------------------------------------------------
1. 시스템에서 직접 Python 환경을 구성할 수 있는 경우 아래와 같이 필요한 패키지를 설치한다.

.. code-block::

  sudo apt install -y python3 python3-pip python-is-python3


2. Python 버전을 확인해 잘 설치되었는지 확인한다.

.. code-block::

  python --version
  Python 3.8.10


.. _InstallPipPackages:

Python SDK 패키지 설치
=======================================

.. tabs::

  .. tab:: PIP를 이용한 설치

    FuriosaAI Python SDK 패키지는 `pypi <https://pypi.org/>`_ 저장소에 업로드 되어 있어
    ``pip`` 명령을 이용하여 다음과 같이 간편하게 설치할 수 있다.

    .. code-block:: sh

      pip install furiosa-sdk


    패키지는 컴파일러 명령행 도구 및 추론 API를 포함하고 있다.
    각각의 자세한 사용법은 :ref:`CompilerCli` 와 :ref:`Tutorial` 를 참고하라.


    추가적인 기능은 Python Extra 패키지 형태로 제공하고 있으며 :ref:`PythonExtraPackages` 에서
    필요한 패키지를 골라 설치할 수 있다. 예를 들어, 모델 서빙을 위해 ``server`` 와 모델과 SDK 간에 호환여부를 확인하기
    위해 ``litmus``를 설치해야 해야 한다면 아래와 같이 확장 패키지를 지정한다.

    .. code-block:: sh

      pip install 'furiosa-sdk[server, litmus]'

  .. tab:: 소스 코드를 이용한 설치

    `FuriosaAI Github 저장소 <https://github.com/furiosa-ai/furiosa-sdk>`_ 에서
    소스코드를 다운 받아 아래와 같은 순서로 설치한다.

    .. code-block:: sh

      git clone https://github.com/furiosa-ai/furiosa-sdk
      cd furiosa-sdk/python
      pip install furiosa-runtime
      pip install furiosa-tools
      pip install furiosa-sdk

    Extra 패키지를 설치하고자 한다면 furiosa-sdk/python 의 서브 디렉토리에 있는 Python
    모듈을 설치하면 된다. 예를 들어, 모델 서버를 설치하고자 한다면 아래와 같이 의존성 순서에 따라
    설치한다.

    .. code-block:: sh

      cd furiosa-sdk/python
      pip install furiosa-registry
      pip install furiosa-server


.. _PythonExtraPackages:

추가 패키지 목록
======================================================

FuriosaAI Models
--------------------------------
NPU에서 바로 실행 가능하며 최적화된 DNN 모델 아키텍쳐와 사전에 훈련된 모델 이미지 등을
Python 모듈 형태로 제공하는 패키지다. 설치는 다음 커맨드로 할 수 있다.

.. code-block:: sh

  pip install 'furiosa-sdk[models]'


Model Server
--------------------------------
DNN 모델을 NPU로 가속하여 GRPC나 Restful API로 서빙하는 기능을 제공한다.
설치는 다음 커맨드로 할 수 있다. 자세한 사용법은 :ref:`ModelServing` 에서 찾을 수 있다.

.. code-block:: sh

  pip install 'furiosa-sdk[server]'


Litmus
--------------------------------
지정한 모델이 FuriosaAI SDK와 호환되는지 여부를 검사하는 도구이다.
이 과정에서 모델 양자화, 컴파일 등의 과정을 모의로 실행한다.

.. code-block:: sh

  pip install 'furiosa-sdk[litmus]'

Quantizer
--------------------------------

Quantizer 패키지는 모델을 양자화(quantization) 된 모델로 변환하기 위한 API 집합을 제공한다.
FuriosaAI SDK와 NPU가 제공하는 양자화 기능에 대한 자세한 내용은 :ref:`ModelQuantization` 에서
찾을 수 있다.

.. code-block:: sh

  pip install 'furiosa-sdk[quantizer]'

