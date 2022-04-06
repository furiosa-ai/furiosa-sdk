.. _Compiler:

****************************************
Compiler
****************************************

Furiosa compiler takes `TFLite <https://www.tensorflow.org/lite>`_ or
`Onnx <https://onnx.ai/>`_ format models as input arguments and compiles
the input models to program binary executable on Furiosa NPU.
During this compilation, the input model is analyzed on operator-level
and the-state-of-the-art optimizations are done for efficient inference
which exploiting Furiosa NPU and resources (CPU and memory) of host machine.
If a model contains only operators on :ref:`SupportedOperators`, then
the model is most likely to be accelerated effectively on Furiosa NPU.


.. _CompilerCli:

``furiosa compile``
-------------------------------------------------
Compiler is mainly called by inference API when initializing Session for prepare inference with model and NPU.
Command ``furiosa compile`` enables user to directly compile input models to executable binaries.
Please refer to :ref:`PythonSDK` for installation of command ``furiosa compile``.

``MODEL_PATH`` is a file path for
`TFLite <https://www.tensorflow.org/lite>`_ or `Onnx <https://onnx.ai/>`_ format file.

.. code-block:: sh

  furiosa compile MODEL_PATH [-o OUTPUT] [--target-npu TARGET_NPU] [--batch BATCH_SIZE]

Option `-o OUTPUT` is optional and the name of output file is determined by the given name.
The default output name is ``output.enf`` when the OUTPUT is not given. Here ENF denotes Executable NPU Format.
The following command generates the ``output.enf`` executable binary file.

.. code-block:: sh

  furiosa compile foo.onnx

The following command generates the ``foo.enf`` executable binary file.

.. code-block::

  furiosa compile foo.onnx -o foo.enf

Option ``--target-npu`` determines the target of NPU for generated executable binary

.. list-table:: Target NPUs
   :widths: 50 50 50
   :header-rows: 1

   * - NPU Family
     - Number of PEs
     - Value
   * - Warboy
     - 1
     - warboy
   * - Warboy
     - 2
     - warboy-2pe

If the generated executable binary is run on single PE of Warboy then following command is useful.

.. code-block::

  furiosa compile foo.onnx --target-npu warboy

If the generated executable binary is run on fused two PEs of Warboy then following command is useful.

.. code-block::

  furiosa compile foo.onnx --target-npu warboy-2pe

Option ``--batch-size`` determines the batch size of input data.
Generally the larger batch size is, the better throughput of inference is obtained.
However, more memory I/O operations are needed when required memory size is over than the size of NPU DRAM. More I/O operations deteriorate overall performance. The default value of ``--batch-size`` is one and the optimal batch can be found by experiments.
For reference, the optimal batch sizes for `MLPerf™ Inference Edge v1.1 <https://mlcommons.org/en/inference-edge-11/>`_ models are like the following:

.. list-table:: Optimal Batch Size for Well-known Models
   :widths: 50 50
   :header-rows: 1

   * - Model
     - Optimal Batch
   * - SSD-MobileNets-v1
     - 2
   * - Resnet50-v1.5
     - 1
   * - SSD-ResNet34
     - 1


The batch size two can be given like the following:

.. code-block::

  furiosa compile foo.onnx --batch-size 2


Usage of ENF files
---------------------------------
File ENF (Executable NPU Format) is the executable binary as the final result of compiler.
Mostly, compilation process takes from several seconds to several minutes. Compilation process can be skipped by using the generated ENF file.

For example, if ENF file is passed to function ``session.create()`` then compilation is skipped and object ``Session`` is instantly created.

.. code-block:: python

  from furiosa.runtime import session
  sess = session.create("foo.enf")