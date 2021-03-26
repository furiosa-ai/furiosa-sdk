********************************************************************
FuriosaAI NPU Python SDK 설치를 위한 Python 3.7+ 환경 구성
********************************************************************


.. _SetupPython:

Python 환경 구성
========================================

Linux 시스템에서 Python 환경 구성
-------------------------------------------------------
시스템에서 바로 Python 환경을 구성할 수 있는 경우 아래와 같이
필요한 패키지를 설치한다.

.. code-block::

  sudo apt install -y python3 python3-pip


.. _AnacondaInstall:

Anaconda를 이용한 Python 환경 구성
-------------------------------------------------------

사용 중인 머신의 Python 환경이 오래된 버전이라 요구사항을 만족하지 못하거나
독립적인 Python 환경이 필요한 경우 `Anaconda <https://docs.conda.io/projects/conda/en/latest/>`_
로 Python 환경을 손쉽게 구성할 수 있다.

.. code-block::

  $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ sh ./Miniconda3-latest-Linux-x86_64.sh
  $ source ~/.bashrc
  $ conda --version
  conda 4.8.3


Anaconda 설치 후에는 독립된 환경을 셋업하고 필요에 따라 활성화 할 수 있다.
FuriosaAI Python SDK는 Python 3.7-3.8 버전과 호환된다. 따라서 python 3.8 새로운 환경을
``furiosa`` 라는 이름으로 생성한다.

.. code-block::

  $ conda create -n furiosa python=3.8


생성한 python 3.8 환경을 ``activate`` 커맨드로 활성화 한다.

.. code-block::

  $ conda activate furiosa
  $ python --version
  Python 3.8.8


그리고 pip 를 이용해 furiosa-sdk를 필요한 extra 패키지를 포함하여 설치한다.

.. code-block::

  $ pip install furiosa-sdk[cli, runtime, quantizer]


생성한 Python 환경의 사용을 비활성화 하고 싶은 경우 ``deactivate`` 커맨드를 사용한다.

.. code-block::

  $ conda deactivate

한번 생성한 환경은 언제든지 다시 ``activate`` 하여 사용할 수 있다.
