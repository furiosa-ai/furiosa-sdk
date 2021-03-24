**********************************
Python SDK 설치
**********************************

Python SDK는 다양한 명령형 실행 도구와 Python 라이브러리를 포함합니다.
선택적으로 필요한 패키지를 설치할 수 있습니다. 


Python SDK 설치를 위해서는 Python 3.6 이상의 환경이 필요하며
시스템에 환경이 준비되어 있지 않은 경우 :ref:`SetupPython` 를 따라 구성해 주세요.
시스템에 직접 Python 환경 구성이 어려운 경우 대안으로 :ref:`AnacondaInstall` 를 시도해보실 수 있습니다.

Python 3.6 이상의 환경이 이미 준비되어 있는 경우 바로 :ref:`SdkInstallation` 
부터 시작하시면 됩니다.

.. _SetupPython:

Python 환경 구성
========================================

Linux 시스템에서 Python 환경 구성
-------------------------------------------------------
시스템에서 바로 Python 환경을 구성할 수 있는 경우 아래와 같이
필요한 패키지를 설치합니다.

.. code-block::
  
  sudo apt install -y python3 python3-pip


.. _AnacondaInstall:

Anaconda를 이용한 Python 환경 구성
-------------------------------------------------------

사용 중인 머신의 Python 환경이 너무 오래된 버전이라 요구사항을 만족하지 못하거나
독립적인 Python 환경이 필요한 경우 `Anaconda <https://docs.conda.io/projects/conda/en/latest/>`_ 
로 Python 환경을 손쉽게 구성할 수 있습니다.

.. code-block::

  $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ sh ./Miniconda3-latest-Linux-x86_64.sh
  $ source ~/.bashrc
  $ conda --version
  conda 4.8.3


Anaconda를 설치 후에는 독립된 환경을 셋업하고 필요에 따라 활성화 할 수 있습니다.
FuriosaAI Python SDK는 Python 3.6-3.8 버전과 호환됩니다. 따라서 python 3.8 환경으로
``furiosa`` 라는 이름으로 새로운 환경을 생성합니다.

.. code-block::

  $ conda create -n furiosa python=3.8


생성한 python 3.8 환경을 ``activate`` 커맨드로 활성화 합니다.

.. code-block::

  $ conda activate furiosa
  $ python --version
  Python 3.8.8


그리고 pip 를 이용해 furiosa-sdk를 필요한 extra 패키지를 포함하여 설치합니다.

.. code-block::

  $ pip install furiosa-sdk[cli, runtime, quantizer]


생성한 Python 환경의 사용을 비활성화 하고 싶은 경우 ``deactivate`` 커맨드를 사용합니다.

.. code-block::

  $ conda deactivate


또한 한번 생성한 환경은 언제든지 다시 ``activate`` 하여 사용할 수 있습니다.




* Python 라이브러리 설치 시에는 :doc:`runtime` 설치가 반드시 필요합니다.

.. _SdkInstallation:

Pypi를 이용한 Python 패키지 설치
========================================

Python SDK는 PIP 커맨드를 이용하여 다음 extra 패키지를 선택적으로 설치할 수 있습니다.

  * cli: 다양한 명령형 도구로 사용 방법은 :doc:`../cli` 를 참고하세요.
  * runtime: 모델 추론을 실행하기 위한 라이브러리를 설치합니다. :doc:`runtime` 가 반드시 필요합니다.
  * analyzer: 모델 분석 도구를 설치합니다.
  * quantizer: 모델의 양자화 도구를 설치합니다.

예를 들어 명령형 도구와 모델 추론을 위한 개발환경이 필요한 경우 아래와 같이 설치합니다.

.. code-block:: sh

  pip install furiosa-sdk[cli, runtime]


어떤 도구가 필요한지 잘 모르는 경우 ``full`` 을 통해 모두 설치해도 좋습니다.

.. code-block:: sh

  pip install furiosa-sdk[full]



Jupyter Notebook 설치
========================================

Jupyter Notebook을 사용하는 경우
FuriosaAI Python SDK와 다양한 Python 에코시스템의 다양한
라이브러리를 편하게 사용할 수 있습니다.

위 설명에 따라 Python SDK를 이미 설치했다면 
pip를 이용해 Jupyter notebook을 간단히 설치해 사용할 수 있습니다.

Jupyter notebook은 아주 다양한 의존된 패키지를 설치하기 때문에
:ref:`AnacondaInstall` 를 이용해 Python 환경을 구성하는 것을 권장합니다.

.. code-block:: sh
  
  $ pip install jupyterlab
  $ jupyter-notebook