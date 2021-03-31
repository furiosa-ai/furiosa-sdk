**********************************
FuriosaAI SDK 설치 사전 준비
**********************************

.. note::

  FuriosaAI SDK는 명령어 인터페이스, 런타임 라이브러리,
  Python 라이브러리를 포함한다. FuriosaAI NPU의 커널 드라이버, 펌웨어 및 런타임은
  FuriosaAI의 평가 프로그램 등록과 최종 사용자 라이센스 동의(End User License Agreement)에 따라
  배포되며, contact@furiosa.ai 로 문의하여 프로그램 다운로드 및 평가를 진행 할 수 있다.


SDK 설치를 위한 최소 요구사항
=====================================================================
* Ubuntu 18.04 LTS (Bionic Beaver) 또는 Debian buster
  또는 상위 버전
* 시스템의 관리자 권한 (root)
* `GitHub <https://github.com/>`_ 및 `PyPi <https://pypi.org/>`_ 로 연결 가능한 네트워크 환경


Linux에서 필수 패키지 설치
=====================================================================

필수 패키지인 ``build-essential`` 와 ``cmake`` 를 설치한다.

.. code-block::

  $ apt-get update
  $ apt-get install cmake build-essential


onnxruntime 1.6.0을 설치한다.
onnxruntime은 `ONNX <https://onnx.ai/>`_ 모델 형식 지원과 모델 양자화를 위해 사용된다.

.. code-block::

  $ wget https://github.com/hyunsik/onnxruntime/releases/download/v1.6.0/libonnxruntime-1.6.0_amd64.deb
  $ apt-get install -y ./libonnxruntime-1.6.0_amd64.deb


.. _SetupPython:

Python SDK 실행 환경 구성
================================================================

FuriosaAI Python SDK 사용은 Python 3.7 또는 그 상위 버전이 필요하다.

.. note::

  FuriosaAI Python SDK를 사용하지 않는다면 이 장을 건너뛰어도 좋다.

.. code-block::

  python --version
  Python 3.8.5

위 명령으로 현재 시스템에 준비되어 있는 Python 버전을 확인할 수 있다.
Python 명령이 존재하지 않거나 하위 버전의 Python을 사용하고 있다면
아래 방법 중에 하나를 선택하여 Python 환경을 구성할 수 있다.

* :ref:`CondaInstall` (권장):
  `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ 는
  특정 Python 어플리케이션만을 위한 전용 Python 환경을 구성할 수 있게 해준다.
  따라서 Conda를 사용하면 Python 어플리케이션을 설치할 때 종종 발생하는 패키지 의존성 문제나 Python 버전 문제를
  피할 수 있다.
* :ref:`SetupPythonOnUbuntu`: Linux 시스템에서 Python 실행환경을 잘 이해하고 있고
  시스템에 직접 Python 환경을 구성하고 싶다면 선택할 수 있다.


.. _CondaInstall:

Conda를 이용한 Python 환경 구성
-------------------------------------------------------

Conda는 특정 Python 어플리케이션만을 위한 전용 Python 환경을 구성할 수 있게 해준다.
Conda에 대해 자세히 알고 싶다면 `Conda`_ 에서 다양한 문서를 참고할 수 있다.


설치 프로그램을 아래와 같이 다운 받아 설치를 시작할 수 있다.
``./Miniconda3-latest-Linux-x86_64.sh`` 실행 시 물어보는 것은 모두 `yes` 를 선택하면 된다.

.. code-block::

  $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ sh ./Miniconda3-latest-Linux-x86_64.sh
  $ source ~/.bashrc
  $ conda --version
  conda 4.8.3


Anaconda 설치 후에는 독립된 Python 실행 환경을 구성하고 필요에 따라 활성화 할 수 있다.
FuriosaAI Python SDK는 Python 3.7-3.8 버전과 호환된다. 따라서 최신 Python 3.8을 이용하여
``furiosa`` 라는 이름으로 실행 환경을 생성하자.

.. code-block::

  $ conda create -n furiosa python=3.8


생성한 Python 3.8 환경은 ``activate`` 명령으로 활성화된다.

.. code-block::

  $ conda activate furiosa
  $ python --version
  Python 3.8.8


그리고 나면 아래처럼 pip 를 이용해 furiosa-sdk를 필요한 extra 패키지와 함께 설치할 수 있다.
자세한 설치 방법은 :doc:`/installation/python-sdk` 를 참고한다.

.. code-block::

  $ pip install furiosa-sdk[cli, runtime]


생성한 Python 환경의 사용을 비활성화하고 싶은 경우 ``deactivate`` 명령을 사용한다.

.. code-block::

  $ conda deactivate

한번 생성한 환경은 언제든지 다시 ``activate`` 하여 사용할 수 있다.
이미 설치했던 패키지는 활성화 후에 다시 설치하지 않아도 된다.


.. _SetupPythonOnUbuntu:

Linux 패키지를 이용한 Python 환경 구성
-------------------------------------------------------
시스템에서 바로 Python 환경을 구성할 수 있는 경우 아래와 같이
필요한 패키지를 설치한다.

.. code-block::

  sudo apt install -y python3 python3-pip
