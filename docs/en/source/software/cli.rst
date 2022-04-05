*********************************************************
Command Line Tools
*********************************************************

Furiosa SDK provides command line tools for printing information of NPU devices, compiling models, and checking compatibility between models and SDK.
This section explains how to install and use the command line tools.

.. _Toolkit:

furiosa-toolkit
===================================
``furiosa-toolkit`` provides command line tools for monitoring and controlling NPU devices.


Installing furiosa-toolkit
--------------------------------------

Kernel driver must be installed by following :ref:`RequiredPackages` before using command line tools.

.. tabs::

  .. tab:: Installation with APT server

    .. code-block:: sh

      sudo apt-get install -y furiosa-toolkit

  .. tab:: Installation with download center

    Download the latest version of the following package.

    * furiosactl

    .. code-block:: sh

      sudo apt-get install -y ./furiosa-toolkit-x.y.z-?.deb


How to use furiosactl
----------------------------------------

Command ``furiosactl`` can be used for checking NPU is recognized well after installing kernel driver.
Currently command ``furiosactl info`` shows NPU device ID, temperature, power, and PCI information.

.. code-block:: sh

  furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu1 | FuriosaAI Warboy |  40°C | 0.00 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+


furiosa
===================================

Command ``furiosa`` is meta command line tool in `Python SDK <PythonSDK>`.
Extra subcommands are added after installing extra packages.
If Python environment is required, then please follow :any:`SetupPython`.

Installing command line tools

.. code-block:: sh

  $ pip install furiosa-sdk


Check installation

.. code-block:: sh

  $ furiosa compile --version
  libnpu.so --- v2.0, built @ fe1fca3
  0.5.0 (rev: 49b97492a built at 2021-12-07 04:07:08) (wrapper: None)


furiosa compile
--------------------

Command ``compile`` compiles `TFLite <https://www.tensorflow.org/lite>`_ or `ONNX <https://onnx.ai/>`_ format models into binary executable on Furiosa NPU. Please refer to :ref:`CompilerCli` for details.

.. _Litmus:

furiosa litmus (checking compatibility of models with Furiosa SDK)
--------------------------------------------

Command ``litmus`` takes `TFLite`_ or `ONNX`_ format models as arguments, and then automatically quantize the models and generate executable binary.
The compatibility of input model with Furiosa SDK is checked through this process.

.. code-block:: sh

  $ furiosa litmus yolov4.onxx
  [Step 1] Checking if the model can be transformed into a quantized model ...
  Quantization: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 85.33it/s]
  [Step 1] Passed
  [Step 2] Checking if the model can be compiled to a NPU program ...
  [Step 2] Passed


If the process fails then error message appears. Please report the error message to
`FuriosaAI Customer Center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_ on
`Bug Report` section for engineering support.

.. code-block:: sh

  $ furiosa litmus efficientnet-lite4-11.onnx

    Stdout:
    [Step 1] Checking if the model can be transformed into a quantized model ...

    Stderr:
    /root/miniconda3/envs/furiosa/lib/python3.8/site-packages/onnx/__init__.py:97: RuntimeWarning: Unexpected end-group tag: Not all data was converted
        decoded = cast(Optional[int], proto.ParseFromString(s))
    [Step 1] Failed
