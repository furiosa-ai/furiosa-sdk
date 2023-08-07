.. _PythonSDK:

****************************************
Python SDK installation and user guide
****************************************

FuriosaAI Python SDK is a software development kit for writing Python applications that use the NPU. 
With the Python SDK, you can utilize various tools, libraries, and frameworks of the Python ecosystem that are most widely used in the AI/ML field for developing NPU applications.
Python SDK consists of various modules and provides an inference API, a quantization API, a command line tool, and a server program for serving.

Requirements 
=======================================================================
* Ubuntu 20.04 LTS (Debian bullseye) or higher
* :ref:`FuriosaAI SDK required packages <RequiredPackages>`
* Python 3.8 or higher (See :any:`SetupPython` for setup Python environment)
* Latest version of pip 

To install and use Python SDK, follow the :ref:`Installing required packages <RequiredPackages>` guide. 
You need to install the required kernel driver, firmware, and runtime library. 

.. _SetupPython:

Python execution environment setup
================================================================

Python SDK requires Python 3.8 or above. Here, we describe Python execution environment configuration.

.. note::

  If you are not using the FuriosaAI Python SDK, or if you are familiar with configuring a Python execution environment, you can skip this section.

You can check the Python version currently installed in your system with the command below.

.. code-block::

  python --version
  Python 3.8.10


If the Python command does not exist, or if your Python version is below 3.8, configure the Python environment
by selecting one of the methods below.

* :ref:`CondaInstall` (recommended):
  `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_ allows you to 
  configure a separate, isolated Python environment for specific Python applications.
  Conda therefore prevents the package dependency issues or Python version issues that users 
  often encounter when installing Python applications. 
* Configure the Python execution environment directly on the :ref:`SetupPythonOnLinux`: Linux system.
  You can select this option if you are not concerned about conflicts with other Python execution environments. 

.. _CondaInstall:

Python environment configuration with Conda
-------------------------------------------------------

Conda makes it easy to configure a isolated Python environment for a specific Python application.
To find out more about Conda, refer to readings available in `Conda`_.

You can get started by downloading the installer as shown below. 
Select `yes` to all questions when running ``sh ./Miniconda3-latest-Linux-x86_64.sh``.

.. code-block::

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh ./Miniconda3-latest-Linux-x86_64.sh
  source ~/.bashrc
  conda --version


Creating and activating isolated Python execution environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After installing Anaconda, you can create an isolated Python execution environment and activate it as needed.

1. If you want to use Python 3.8, create an execution environment with the name ``furiosa-3.8``, by using the following command.

.. code-block::

  conda create -n furiosa-3.8 python=3.8


2. The created Python 3.8 environment is activated with the ``activate`` command.

.. code-block::

  conda activate furiosa-3.8
  # version check 
  python --version


3. Once the Python execution environment is activated, install the Python SDK as explained in :ref:`InstallPipPackages`.


4. If you wish to terminate the Python execution environment, use the ``deactivate`` command.

.. code-block::

  $ conda deactivate

An environment created once can be used again at any time with the ``activate`` command. 
Packages that have been installed do not need to be reinstalled after activation. 


.. _SetupPythonOnLinux:

Configuring Python environment using Linux packages 
-------------------------------------------------------
1. If you can configure the Python environment directly on the system, install the necessary packages as shown below. 

.. code-block::

  sudo apt install -y python3 python3-pip python-is-python3


2. Check the Python version to ensure proper installation. 

.. code-block::

  python --version
  Python 3.8.10


.. _InstallPipPackages:

Installing Python SDK package 
=======================================

Before installing the furiosa-sdk, you need to update Python's package installer to the latest version.

.. code-block:: sh

    pip install --upgrade pip setuptools wheel


.. warning::

  If you install the furiosa-sdk without updating to the latest version, you may encounter the following error.

  .. code-block:: sh
    
      ERROR: Could not find a version that satisfies the requirement furiosa-quantizer-impl==0.10.* (from furiosa-quantizer==0.10.*->furiosa-sdk) (from versions: none)
      ERROR: No matching distribution found for furiosa-quantizer-impl==0.10.* (from furiosa-quantizer==0.10.*->furiosa-sdk)


.. tabs::

  .. tab:: Installing with PIP

    FuriosaAI Python SDK package is uploaded on the `pypi <https://pypi.org/>`_ repository, 
    so you can easily install it as shown by using the ``pip`` command.

    .. code-block:: sh

      pip install furiosa-sdk

    The package contains a compiler command line interface and an inference API.
    Refer to :ref:`CompilerCli` and :ref:`Tutorial` for detailed usage guides.

    Additional functions are provided in the form of Python extra packages, and you can select and
    install packages as you require from :ref:`PythonExtraPackages`.
    For example, if you need to install `server`` for model serving and 
    ``litmus`` to check the compatibility between model and SDK, specify the extension package as follows.

    .. code-block:: sh

      pip install 'furiosa-sdk[server, litmus]'

  .. tab:: Installing with the source code 

    Download the source code from `FuriosaAI Github repository <https://github.com/furiosa-ai/furiosa-sdk>`_ 
    and install the packages in the following order.

    .. code-block:: sh

      git clone https://github.com/furiosa-ai/furiosa-sdk
      cd furiosa-sdk/python
      pip install furiosa-runtime
      pip install furiosa-tools
      pip install furiosa-sdk

    If you wish to install extra packages, install the Python module in the subdirectory of furiosa-sdk/python. 
    For example, if you want to install a model server, install it according to the order of dependencies as follows. 

    .. code-block:: sh

      cd furiosa-sdk/python
      pip install furiosa-server


.. _PythonExtraPackages:

Extra packages
======================================================

Legacy Runtime/API
--------------------------------
Rather than the next-generation runtime and its API newly adoted since 0.10.0,
you can install furiosa-sdk with the legacy runtime and API as following:

.. code-block:: sh

  pip install 'furiosa-sdk[legacy]'


FuriosaAI Models
--------------------------------
It can be executed directly on the NPU and provides optimized DNN model architecture, pre-trained 
model image, among others, in the form of a Python module. 
You can install them with the following command. 

.. code-block:: sh

  pip install 'furiosa-sdk[models]'


Quantizer
--------------------------------
The quantizer package provides a set of APIs for converting a model into a quantized model.
You can find more information about the quantization function provided by the Furiosa SDK and the NPU
at :ref:`ModelQuantization`.

.. code-block:: sh

  pip install 'furiosa-sdk[quantizer]'


Model Server
--------------------------------
Provides the function of accelerating DNN model with the NPU, and serving it with GRPC or Restful API.

.. code-block:: sh

  pip install 'furiosa-sdk[server]'

Litmus
--------------------------------
A tool to check whether the specified model is compatible with the Furiosa SDK. 
Here, we simulate execution of processes such as model quantization and compilation. 

.. code-block:: sh

  pip install 'furiosa-sdk[litmus]'

