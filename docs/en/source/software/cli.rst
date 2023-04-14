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



furiosactl instructions
----------------------------------------
The furiosactl command provides a variety of subcommands and has the ability to obtain information or control the device.

.. code-block:: sh

    furiosactl <sub command> [option] ..


``furiosactl info``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After installing the kernel driver, you can use the ``furiosactl`` command to check whether the NPU device is recognized.
Currently, this command provides the ``furiosactl info`` command to output temperature, power consumption and PCI information of the NPU device.
If the device is not visible with this command after mounting it on the machine, :ref:`RequiredPackages` to install the driver.
If you add the ``--full`` option to the ``info`` command, you can see the device's UUID and serial number information together.


.. code-block:: sh

  $ furiosactl info
  +------+--------+----------------+-------+--------+--------------+
  | NPU  | Name   | Firmware       | Temp. | Power  | PCI-BDF      |
  +------+--------+----------------+-------+--------+--------------+
  | npu1 | warboy | 1.6.0, 3c10fd3 |  54°C | 0.99 W | 0000:44:00.0 |
  +------+--------+----------------+-------+--------+--------------+

  $ furiosactl info --full
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+
  | NPU  | Name   | UUID                                 | S/N               | Firmware       | Temp. | Power  | PCI-BDF      | PCI-DEV |
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+
  | npu1 | warboy | 00000000-0000-0000-0000-000000000000 | WBYB0000000000000 | 1.6.0, 3c10fd3 |  54°C | 0.99 W | 0000:44:00.0 | 511:0   |
  +------+--------+--------------------------------------+-------------------+----------------+-------+--------+--------------+---------+

``furiosactl list``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``list`` subcommand provides information about the device files available on the NPU device.
You can also check whether each core present in the NPU is in use or idle.

.. code-block:: sh

  furiosactl list
  +------+------------------------------+-----------------------------------+
  | NPU  | Cores                        | DEVFILES                          |
  +------+------------------------------+-----------------------------------+
  | npu1 | 0 (available), 1 (available) | npu1, npu1pe0, npu1pe1, npu1pe0-1 |
  +------+------------------------------+-----------------------------------+


``furiosactl ps``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``ps`` subcommand prints information about the OS process currently occupying the NPU device.

.. code-block:: sh

    $ furiosactl ps
    +-----------+--------+------------------------------------------------------------+
    | NPU       | PID    | CMD                                                        |
    +-----------+--------+------------------------------------------------------------+
    | npu0pe0-1 | 132529 | /usr/bin/python3 /usr/local/bin/uvicorn image_classify:app |
    +-----------+--------+------------------------------------------------------------+


``furiosactl top`` (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``top`` subcommand is used to view utilization by NPU unit over time.
The output has the following meaning
By default, utilization is calculated every 1 second, but you can set the calculation interval yourself with the ``--interval`` option. (unit: ms)

.. list-table:: furiosa top fields
   :widths: 100 400
   :header-rows: 1

   * - Item
     - Description
   * - Datetime
     - Observation time
   * - PID
     - Process ID that is using the NPU
   * - Device
     - NPU device in use
   * - NPU(%)
     - Percentage of time the NPU was used during the observation time.
   * - Comp(%)
     - Percentage of time the NPU was used for computation during the observation time
   * - I/O (%)
     - Percentage of time the NPU was used for I/O out of the time the NPU was used
   * - Command
     - Executed command line of the process


.. code-block:: sh

    $ furiosactl top --interval 200
    NOTE: furiosa top is under development. Usage and output formats may change.
    Please enter Ctrl+C to stop.
    Datetime                        PID       Device        NPU(%)   Comp(%)   I/O(%)   Command
    2023-03-21T09:45:56.699483936Z  152616    npu1pe0-1      19.06    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:56.906443888Z  152616    npu1pe0-1      51.09     93.05     6.95   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.110489333Z  152616    npu1pe0-1      46.40     97.98     2.02   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.316060982Z  152616    npu1pe0-1      51.43    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.521140588Z  152616    npu1pe0-1      54.28     94.10     5.90   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.725910558Z  152616    npu1pe0-1      48.93     98.93     1.07   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:57.935041998Z  152616    npu1pe0-1      47.91    100.00     0.00   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf
    2023-03-21T09:45:58.13929122Z   152616    npu1pe0-1      49.06     94.94     5.06   ./npu_runtime_test -n 10000 results/ResNet-CTC_kor1_200_nightly3_128dpes_8batches.enf


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

  $ furiosa litmus yolov4.onnx
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
