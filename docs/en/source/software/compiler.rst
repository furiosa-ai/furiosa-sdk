.. _Compiler:

****************************************
Compiler 
****************************************
The FuriosaAI compiler compiles models of formats `TFLite <https://www.tensorflow.org/lite>`_ and `Onnx <https://onnx.ai/>`_ model ((`OpSet 13 <https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-13-of-the-default-onnx-operator-set>`_ or lower version), thereby generating programs that execute inference using FuriosaAI NPU and resources (CPU, memory, etc) of the host machine.
In this process, the compiler analyses the model at the operator level, optimizes it, and generates a program so as to maximize NPU acceleration and host resources utilization. Even for models that are not well known,
so long as supported operators are utilized well, you can design models that are optimized for the NPU . 

You can find the list of NPU acceleration supported operators at :ref:`SupportedOperators`.

.. _CompilerCli:

``furiosa-compiler``
-------------------------------------------------
The most common ways to use a compiler would be to automatically call it 
during the process of resetting the inference API or the NPU.  

But you can directly compile a model and generate a program by using the command line tool ``furiosa-compiler`` in shell. You can install ``furiosa-compiler`` command via APT package manager.

.. code-block:: sh
  
  $ apt install furiosa-compiler


The usage of ``furiosa-compiler`` is as follows:

.. code-block:: sh

  $ furiosa-compiler --help
  
  Furiosa SDK Compiler v0.10.0 (f8f05c8ea 2023-07-31T19:30:30Z)

  Usage: furiosa-compiler [OPTIONS] <SOURCE>

  Arguments:
    <SOURCE>
            Path to source file (tflite, onnx, and other IR formats, such as dfg, cdfg, gir, lir)

  Options:
    -o, --output <OUTPUT>
            Writes output to <OUTPUT>
            
            [default: output.<TARGET_IR>]

    -b, --batch-size <BATCH_SIZE>
            Specifies the batch size which is effective when SOURCE is TFLite, ONNX, or DFG

        --target-ir <TARGET_IR>
            (experimental) Target IR - possible values: [enf]
            
            [default: enf]

        --target-npu <TARGET_NPU>
            Target NPU family - possible values: [warboy, warboy-2pe]
            
            [default: warboy-2pe]

        --dot-graph <DOT_GRAPH>
            Filename to write DOT-formatted graph to

        --analyze-memory <ANALYZE_MEMORY>
            Analyzes the memory allocation and save the report to <ANALYZE_MEMORY>

    -v, --verbose
            Shows details about the compilation process

        --no-cache
            Disables the compiler result cache

    -h, --help
            Print help (see a summary with '-h')

    -V, --version
            Print version


``SOURCE`` is the file path of 
`TFLite <https://www.tensorflow.org/lite>`_ or `ONNX <https://onnx.ai/>`_.
You have to use quantized models through :ref:`ModelQuantization` for NPU accleration.

You can omit the option `-o OUTPUT`, and you can also choose to designate the output file name.
When omitted, the default output file name is ``output.enf``. Here, enf stands for Executable NPU Format.
So if you run as shown below, it will generate a ``output.enf`` file.

.. code-block:: sh

  furiosa-compiler foo.onnx

If you designate the output file name as below, it will generate a ``foo.enf`` file.

.. code-block::

  furiosa-compiler foo.onnx -o foo.enf

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

  furiosa-compiler foo.onnx --target-npu warboy

When 2 PEs are fused, execute as follows.

.. code-block::

  furiosa-compiler foo.onnx --target-npu warboy-2pe

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

  furiosa-compiler foo.onnx --batch-size 2


Using ENF files
---------------------------------
After the compilation process, the final output of the FuriosaAI compiler is ENF (Executable NPU Format) type data. 
In general, the compilation process takes from a few seconds to several minutes depending on the model. 
Once you have the ENF file, you can reuse it to omit this compilation process. 

This may be useful if you need to frequently create sessions or
serve one model across several machines in an actual operation environment.

For example, you can first create an ENF file by referring to :ref:`CompilerCli`.
Then, with :ref:`PythonSDK <PythonSDK>` as shown below,
you can instantly create a runner without the compilation process by
passing the ENF file as an argument to the ``create_runner()`` function as follows:

.. code-block:: python
  
  from furiosa.runtime import sync
  
  with sync.create_runner("path/to/model.enf") as runner:
    outputs = runner.run(inputs)


.. _CompilerCache:

Compiler Cache
-------------------------------------------
Compiler cache allows to user applications to reuse once-compiled results.
It's very helpful especially when you are developing applications because the compilation
usually takes at least a couple of minutes.

By default, the compiler cache uses a local file system (``$HOME/.cache/furiosa/compiler``) as a cache storage.
If you specify a configuration, you can also use Redis as a remote and distributed cache storage.

The compiler cache is enabled by default, but you can explicitly enable or disable the cache by setting ``FC_CACHE_ENABLED``.
This setting is effective in CLI tools, Python SDK, and serving frameworks.

.. code-block:: sh

  # Enable Compiler Cache
  export FC_CACHE_ENABLED=1
  # Disable Compiler Cache
  export FC_CACHE_ENABLED=0

The default cache location is ``$HOME/.cache/furiosa/compiler``, but you can explicitly specify the cache storage
by setting the shell environment variable ``FC_CACHE_STORE_URL``. If you want to Redis as a cache storage,
you can specify some URLs starting with ``redis://`` or ``rediss://`` (over SSL).

.. code-block:: sh

  # When you want to specify a cache directory
  export FC_CACHE_STORE_URL=/tmp/cache

  # When you want to specify a Redis cluster as the cache storage
  export FC_CACHE_STORE_URL=redis://:<PASSWORD>@127.0.0.1:6379
  # When you want to specify a Redis cluster over SSL as the cache storage
  export FC_CACHE_STORE_URL=rediss://:<PASSWORD>@127.0.0.1:25945

The cache will be valid for 30 days by default, but you can explicitly specify the cache lifetime by setting
seconds to the environment variable ``FC_CACHE_LIFETIME``.

.. code-block:: sh

  # 2 hours cache lifetime
  export FC_CACHE_LIFETIME=7200

Also, you can control more the cache behavior according to your purpose as following:

.. list-table:: Cache behaviors according to ``FC_CACHE_LIFETIME``
   :widths: 50 200 50
   :header-rows: 1

   * - Value (secs)
     - Description
     - Example
   * - *N* > 0
     - Cache will be alive for N secs
     - 7200 (2 hours)
   * - 0
     - All previous cache will be invalidated. (When you want to compile the model without cache)
     - 0
   * - *N* < 0
     - Cache will be alive forever without expiration. (it can be useful when you want read-only cache)
     - -1
