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

The ``litmus`` is a tool to check quickly if an `ONNX`_ model can work normally with Furiosa SDK using NPU.
``litmus`` goes through all usage steps of Furiosa SDK, including quantization, compilation, and inferences on FuriosaAI NPU.
``litmus`` is also a useful bug reporting tool. If you specify ``--dump`` option, ``litmus`` will collect logs and environment information and dump an archive file.
The archive file can be used to report issues.

The steps executed by ``litmus`` command are as follows.

  - Step1: Load an input model and check it is a valid model.
  - Step2: Quantize the model with random calibration.
  - Step3: Compile the quantized model.
  - Step4: Inference the compiled model using ``furiosa-bench``. This step is skipped if ``furiosa-bench`` was not installed.


Usage:

.. code-block:: sh

  furiosa-litmus [-h] [--dump OUTPUT_PREFIX] [--skip-quantization] [--target-npu TARGET_NPU] [-v] model_path

A simple example using ``litmus`` command is as follows.

.. code-block:: sh

  $ furiosa litmus model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Checking if the model can be loaded and optimized ...
  [Step 1] Passed
  [Step 2] Checking if the model can be quantized ...
  [Step 2] Passed
  [Step 3] Checking if the model can be compiled for the NPU family [warboy-2pe] ...
  [1/6] 🔍   Compiling from onnx to dfg
  Done in 0.09272794s
  [2/6] 🔍   Compiling from dfg to ldfg
  ▪▪▪▪▪ [1/3] Splitting graph(LAS)...Done in 9.034934s
  ▪▪▪▪▪ [2/3] Lowering graph(LAS)...Done in 20.140083s
  ▪▪▪▪▪ [3/3] Optimizing graph...Done in 0.019548794s
  Done in 29.196825s
  [3/6] 🔍   Compiling from ldfg to cdfg
  Done in 0.001701888s
  [4/6] 🔍   Compiling from cdfg to gir
  Done in 0.015205072s
  [5/6] 🔍   Compiling from gir to lir
  Done in 0.0038304s
  [6/6] 🔍   Compiling from lir to enf
  Done in 0.020943863s
  ✨  Finished in 29.331545s
  [Step 3] Passed
  [Step 4] Perform inference once for data collection... (Optional)
  ✨  Finished in 0.000001198s
  ======================================================================
  This benchmark was executed with latency-workload which prioritizes latency of individual queries over throughput.
  1 queries executed with batch size 1
  Latency stats are as follows
  QPS(Throughput): 125.00/s

  Per-query latency:
  Min latency (us)    : 7448
  Max latency (us)    : 7448
  Mean latency (us)   : 7448
  50th percentile (us): 7448
  95th percentile (us): 7448
  99th percentile (us): 7448
  99th percentile (us): 7448
  [Step 4] Finished


If you have quantized model already, you can skip Step1 and Step2 with ``--skip-quantization`` option.


.. code-block:: sh

  $ furiosa litmus --skip-quantization quantized-model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Skip model loading and optimization
  [Step 2] Skip model quantization
  [Step 1 & Step 2] Load quantized model ...
  [Step 3] Checking if the model can be compiled for the NPU family [warboy-2pe] ...
  ...


You can use the ``--dump <path>`` option to create a `<path>-<unix_epoch>.zip` file that contains metadata necessary for analysis, such as compilation logs, runtime logs, software versions, and execution environments.
If you have any problems, you can get support through `FuriosaAI customer service center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_ with this zip file.


.. code-block:: sh

  $ furiosa litmus --dump archive model.onnx
  libfuriosa_hal.so --- v0.11.0, built @ 43c901f
  INFO:furiosa.common.native:loaded native library libfuriosa_compiler.so.0.10.0 (0.10.0-dev d7548b7f6)
  furiosa-quantizer 0.10.0 (rev. 9ecebb6) furiosa-litmus 0.10.0 (rev. 9ecebb6)
  [Step 1] Checking if the model can be loaded and optimized ...
  [Step 1] Passed
  ...

  $ zipinfo -1 archive-1690438803.zip 
  archive-16904388032l4hoi3h/meta.yaml
  archive-16904388032l4hoi3h/compiler/compiler.log
  archive-16904388032l4hoi3h/compiler/memory-analysis.html
  archive-16904388032l4hoi3h/compiler/model.dot
  archive-16904388032l4hoi3h/runtime/trace.json
