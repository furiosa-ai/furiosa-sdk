*********************************************************
Command Line Tools
*********************************************************

Through the command line tools, Furiosa SDK provides functions such as monitoring NPU device information, compiling models, and checking compatibility between models and SDKs. This section explains how to install and use each command line tool.

.. _Toolkit:

furiosa-toolkit
===================================
``furiosa-toolkit`` provides a command line tool that enables users to manage and check the information of NPU devices.


furiosa-toolkit installation
--------------------------------------
To use this command line tool, you first need to install the kernel driver as shown in :ref:`RequiredPackages`.
Subsequently, follow the instructions below to install furiosa-toolkit.

.. tabs::

  .. tab:: Installation using APT server

    .. code-block:: sh

      sudo apt-get install -y furiosa-toolkit

  .. tab:: Installation using download center

    Select and download the latest versions of the packages listed below. Install them in order as written in the command.

    * furiosactl

    .. code-block:: sh

      sudo apt-get install -y ./furiosa-toolkit-x.y.z-?.deb


furiosactl instructions
----------------------------------------

After installing the kernel driver, you can use the ``furiosactl`` command to check whether the NPU device is recognized.
Currently, this command provides the ``furiosactl info`` command to output the Device ID, temperature, power consumption and PCI information of the NPU device.


.. code-block:: sh

  furiosactl info
  +------+------------------+-------+--------+--------------+---------+
  | NPU  | Name             | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+------------------+-------+--------+--------------+---------+
  | npu1 | FuriosaAI Warboy |  40°C | 0.00 W | 0000:01:00.0 | 509:0   |
  +------+------------------+-------+--------+--------------+---------+


furiosa
===================================

The ``furiosa`` command is a meta-command line tool that can be used by installing the `Python SDK <PythonSDK>`.
Additional subcommands are also added when the extension package is installed.

If the Python execution environment is not prepared, refer to :any:`SetupPython`.


Installing command line tool.

.. code-block:: sh

  $ pip install furiosa-sdk


Verifying installation.

.. code-block:: sh

  $ furiosa compile --version
  libnpu.so --- v2.0, built @ fe1fca3
  0.5.0 (rev: 49b97492a built at 2021-12-07 04:07:08) (wrapper: None)


furiosa compile
--------------------

The ``compile`` command compiles models such as `TFLite <https://www.tensorflow.org/lite>`_ and `ONNX <https://onnx.ai/>`_, generating programs that utilize FuriosaAI NPU.

Detailed explanations and options can be found in the :ref:`CompilerCli` page.

.. _Litmus:

furiosa litmus (Checking for model compatibility)
----------------------------------------------------------------------

The ``litmus`` command takes the `TFLite`_ and `ONNX`_ models as arguments,
quantizes them automatically, and attempts to compile up to the final binary, in order to check whether the given model is compatible with the SDK.

.. code-block:: sh

  $ furiosa litmus yolov4.onxx
  [Step 1] Checking if the model can be transformed into a quantized model ...
  Quantization: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 85.33it/s]
  [Step 1] Passed
  [Step 2] Checking if the model can be compiled to a NPU program ...
  [Step 2] Passed


Should it fail, you will see an error message like the one below. You can seek help by filing a bug report to
`FuriosaAI customer service center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_.

.. code-block:: sh

  $ furiosa litmus efficientnet-lite4-11.onnx

    Stdout:
    [Step 1] Checking if the model can be transformed into a quantized model ...

    Stderr:
    /root/miniconda3/envs/furiosa/lib/python3.8/site-packages/onnx/__init__.py:97: RuntimeWarning: Unexpected end-group tag: Not all data was converted
        decoded = cast(Optional[int], proto.ParseFromString(s))
    [Step 1] Failed
