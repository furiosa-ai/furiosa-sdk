.. _PythonSDK:

*********************************************
Python SDK Installation & User Guide
*********************************************

Using the Furiosa Python SDK library, you can easily write Python application programs that facilitate the NPU.
Python SDK makes application programmers to exploit various toolkits, libraries, and frameworks widely used in Python ecosystem.
Pythoon SDK consists of several modules and provides inference API, quantization API, command line interface tools, and inference serving.

Requirements for installation
=======================================================================
* Ubuntu 18.04 LTS (Debian buster) or higher version
* :ref:`FuriosaAI SDK Required Packages <RequiredPackages>`
* Python 3.7 or higher version (please refer to :any:`SetupPython` for setting up Python environment)
* pip latest version

In order to use Python SDK, kernel driver, firmware, and runtime library must be installed (refer to :ref:`FuriosaAI SDK Required Packages <RequiredPackages>` for installation)

.. _SetupPython:

Setting up Python Environment
================================================================

Python SDK requires Python 3.7 or higher version. This section introduces how to set up Python environment.

.. note::
    Please skip this chapter either FuriosaAI Python SDK is not required or you are used to Python environment.

You may check the version of Python on system by the following command:

.. code-block::

  python --version
  Python 3.8.10

If Python command is not installed yet or Python version is lower than 3.8, Python environment can be setup by choosing one of options below:

* :ref:`CondaInstall` (recommended):
  `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ provides an isolated Python environment for certain applications.
  Conda resolves some problems such as package dependency conflicts and version conflicts when installing Python applications.
* :ref:`SetupPythonOnLinux` - setting up Python environment directly on Linux:
  Recommended only when there are no conflicts with other Python applications.

.. _CondaInstall:

Setting up Python Environment with Conda
-------------------------------------------------------

Conda provides an isolated Python environment for certain applications.
Please refer to `Conda`_ for details.

Installation requires download of Conda program.
``sh ./Miniconda3-latest-Linux-x86_64.sh`` choose `yes` for all questions.

.. code-block::

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh ./Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  conda --version


Creating and Activating Isolated Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An isolated Python environment can be created and activated after installing Anaconda.

1. Create an isolated Python environment named ``furiosa-3.8`` which uses Python version 3.8.

.. code-block::

  conda create -n furiosa-3.8 python=3.8


2. Activate the created Python 3.8 environment with command ``activate``.

.. code-block::

  conda activate furiosa-3.8
  # check the version of Python
  python --version


3. Install Python SDK with instructions on :ref:`InstallPipPackages`.


4. Deactivate the Python environment with command ``deactivate``.

.. code-block::

  $ conda deactivate

The created Python environment can be activated again with command ``activate``.
The installed Python SDK remains on the Python environment hence the Python SDK does not need to be installed again.


.. _SetupPythonOnLinux:

Setting up Python Environment on Linux
-------------------------------------------------------
1. Install the required packages for Python environment.

.. code-block::

  sudo apt install -y python3 python3-pip python-is-python3


2. Check the version of Python.

.. code-block::

  python --version
  Python 3.8.10


.. _InstallPipPackages:

Installation of Python SDK Package
=======================================

.. tabs::

  .. tab:: installation with PIP

    FuriosaAI Python SDK package is uploaded on PyPi storage `pypi <https://pypi.org/>`_, hence it can be easily installed with command ``pip``.

    .. code-block:: sh

      pip install furiosa-sdk


    Package contains compiler command line tools and inference API.
    Please refer to :ref:`CompilerCli` and :ref:`Tutorial` for details.

    Extra Python packages contains other extra functions, please refer to :ref:`PythonExtraPackages`.
    Furiosa ``litmus`` can check whether user provided models are compatible with Furiosa NPU.

    .. code-block:: sh

      pip install 'furiosa-sdk[litmus]'

  .. tab:: installation with source code

    Download `FuriosaAI Github Repository <https://github.com/furiosa-ai/furiosa-sdk>`_ and install Furiosa SDK with the following instructions.

    .. code-block:: sh

      git clone https://github.com/furiosa-ai/furiosa-sdk
      cd furiosa-sdk/python
      pip install furiosa-common
      pip install furiosa-tools
      pip install furiosa-runtime
      pip install furiosa-sdk

    In order to install extra packages, please install the modules on the sub directories of furiosa-sdk/python.
    For example, model server can be installed on the following dependency.

    .. code-block:: sh

      cd furiosa-sdk/python
      pip install furiosa-registry
      pip install furiosa-server


.. _PythonExtraPackages:

Extra Packages
======================================================

FuriosaAI Models
--------------------------------
FuriosaAI Models contain pre-trained DNN models which are optimized on Furiosa NPU.

.. code-block:: sh

  pip install 'furiosa-sdk[models]'

Model Server
--------------------------------
Model Server provides GRPC and Restful API of model inference accelerated on Furiosa NPU.
Please refer to :ref:`ModelServing` for details.

.. code-block:: sh

  pip install 'furiosa-sdk[server]'


Litmus
--------------------------------
Litmus checks that the user-provided model is compatible with FuriosaAI SDK.
The user-provided models are quickly quantized and compiled to binary for checking compatibility.

.. code-block:: sh

  pip install 'furiosa-sdk[litmus]'

Quantizer
--------------------------------

Quantizer package provides quantization API. Please refer to :ref:`ModelQuantization` for details.

.. code-block:: sh

  pip install 'furiosa-sdk[quantizer]'

