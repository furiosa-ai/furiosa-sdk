********************************************
FuriosaAI SDK Installation + Prerequisites
********************************************

.. note::

  The FuriosaAI SDK is composed of a commmand line interface, runtime library, and Python library. 
  The FuriosaAI NPU's kernel driver, firmware, and runtime are distributed according to 
  FuriosaAI's evaluation tool registration and End User License Agreement. 
  For questions about downloading and evaluating the program, please contact us at contact@furiosa.ai.

Minimum Requirements 
=====================================================================
* Ubuntu 18.04 LTS (Bionic Beaver), Debian buster, or later
* System administrator privileges (root)
* A Network Environment/Internet connection able to connect to `GitHub <https://github.com/>`_ and `PyPi <https://pypi.org/>`_ 


Installing Dependencies (Linux)
=====================================================================

To install ``build-essential`` and ``cmake``:

.. code-block::

  $ apt-get update
  $ apt-get install cmake build-essential


To install ``onnxruntime`` 1.8.1:

``onnxruntime`` is used for `ONNX <https://onnx.ai/>`_ model format support and model quantization.

.. code-block::

  $ wget https://github.com/hyunsik/onnxruntime/releases/download/v1.8.1/libonnxruntime-1.8.1_amd64.deb
  $ apt-get install -y ./libonnxruntime-1.8.1_amd64.deb


.. _SetupPython:

Configuring the Python SDK Runtime Environment
================================================================

Python 3.7 or later is required to use the FuriosaAI Python SDK.

.. note::

  If you aren't using the FuriosaAI Python SDK, you can skip this section.

.. code-block::

  python --version
  Python 3.8.5

The above command checks the version of Python available on the current system.
If your system doesn't have Python or only has a lower version of it, 
you can configure a Python environment with one of the following methods.

* :ref:`CondaInstall` (recommended):
  `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ can 
  be used to configure a dedicated Python environment for specific Python applications. 
  Conda helps to prevent package dependency issues and Python version issues
  that can occur when installing complicated Python applications.
* :ref:`SetupPythonOnUbuntu`: This option is for users who have a good understanding 
  of the Python execution environment on their Linux system and want to configure 
  the Python environment directly on their system.

.. _CondaInstall:

Configuring the Python Environment Using Conda
-------------------------------------------------------

You can use Conda to configure a dedicated Python environment for specific Python applications.
Please refer to the documents in `Conda`_ for more information on Conda.


Start the installation process by downloading the installation program as follows:
Run ``./Miniconda3-latest-Linux-x86_64.sh`` and select `yes`.

.. code-block::

  $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ sh ./Miniconda3-latest-Linux-x86_64.sh
  $ source ~/.bashrc
  $ conda --version
  conda 4.8.3


After installing Anaconda, you can configure an isolated Python runtime environment 
and activate it as needed. The FuriosaAI Python SDK is compatible with Python 3.7-3.9 versions.
The following uses Python 3.8 version to create an execution environment named ``furiosa``.

.. code-block::

  $ conda create -n furiosa python=3.8


The newly created Python 3.8 environment is activated with the ``activate`` command.

.. code-block::

  $ conda activate furiosa
  $ python --version
  Python 3.8.8


Now you can install furiosa-sdk with additional dependencies using pip as shown below.
For more detailed instructions for installation, refer to  :doc:`/installation/python-sdk`.

.. code-block::

  $ pip install 'furiosa-sdk[cli, runtime]'


To deactivate the use of the user created Python environment, use the ``deactivate`` command.

.. code-block::

  $ conda deactivate

Once created, the environment can be reactivated using ``activate`` and used at any time. 
Packages that have already been installed do not need to be reinstalled after activation.


.. _SetupPythonOnUbuntu:

Configuring the Python Environment Using Linux Packages
-------------------------------------------------------
If a Python environment can be configured directly from the system, 
install the following dependencies as shown below. 

.. code-block::

  sudo apt install -y python3 python3-pip
