.. _Compiler:

****************************************
Compiler 
****************************************
The FuriosaAI compiler compiles models of formats `TFLite <https://www.tensorflow.org/lite>`_ and `Onnx <https://onnx.ai/>`_, thereby generating programs that execute inference using FuriosaAI NPU and resources (CPU, memory, etc) of the host machine.
In this process, the compiler analyses the model at the operator level, optimizes it, and generates a program so as to maximize NPU acceleration and host resources utilization. Even for models that are not well known,
so long as supported operators are utilized well, you can design models that are optimized for the NPU . 

You can find the list of NPU acceleration supported operators at :ref:`SupportedOperators`.

.. _CompilerCli:

``furiosa compile``
-------------------------------------------------
The most common ways to use a compiler would be to automatically call it 
during the process of resetting the inference API or the NPU.  

But you can directly compile a model and generate a program by using the command line tool ``furiosa compile`` in shell. You can use the ``furiosa compile`` command by installing :ref:`PythonSDK`.

The arguments of the command are as follows. ``MODEL_PATH`` is the file path of 
`TFLite <https://www.tensorflow.org/lite>`_ or `Onnx <https://onnx.ai/>`_.

.. code-block:: sh

  furiosa compile MODEL_PATH [-o OUTPUT] [--target-npu TARGET_NPU] [--batch BATCH_SIZE]

You can omit the option `-o OUTPUT`, and you can also choose to designate the output file name.
When omitted, the default output file name is ``output.enf``. Here, enf stands for Executable NPU Format.
So if you run as shown below, it will generate a ``output.enf`` file.

.. code-block:: sh

  furiosa compile foo.onnx

If you designate the output file name as below, it will generate a ``foo.enf`` file.

.. code-block::

  furiosa compile foo.onnx -o foo.enf

``--target-npu`` lets the generated binary to designate target NPU는.

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

If generated program's target NPU is Warboy that uses one PE independently, you can run the following command. 

.. code-block::

  furiosa compile foo.onnx --target-npu warboy

When 2 PEs are fused, execute as follows.

.. code-block::

  furiosa compile foo.onnx --target-npu warboy-2pe

The ``--batch-size`` option lets you specify `batch size`, the number of samples 
to be passed as input when executing inference through the inference API. 
The larger the batch size, the higher the NPU utilization, since more data is given as input and executed
at once. This allows the inference process to be shared across the batch, increasing efficiency. 
However, if the larger batch size results in the necessary memory size exceeding NPU DRAM size, 
the memory I/O cost between the host and the NPU may increase and lead to significant performance degradation. 
The default value of batch size is one. Appropriate value can usually be found through trial and error.
For reference, the optimal batch sizes for some models included in the 
`MLPerf™ Inference Edge v2.0 <https://mlcommons.org/en/inference-edge-20/>`_ benchmark are as follows.

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

If your desired batch size is two, you can run the following command.

.. code-block::

  furiosa compile foo.onnx --batch-size 2


Using ENF files
---------------------------------
After the compilation process, the final output of the FuriosaAI compiler is ENF (Executable NPU Format) type data. 
In general, the compilation process takes from a few seconds to several minutes depending on the model. 
Once you have the ENF file, you can reuse it to omit this compilation process. 

For example, when using ref:`PythonSDK <PythonSDK>` as shown below, 
if you pass an ENF file as an argument to the ``session.create()`` function, you may skip the compiling process and 
immediately generate the ``Session`` object. 

.. code-block:: python

  from furiosa.runtime import session
  sess = session.create("foo.enf")