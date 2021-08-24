**********************************************
Python SDK Quickstart
**********************************************

Using the FuriosaAI NPU Python SDK library, you can easily write Python programs that leverage the NPU.


**Minimum Requirements**

* :doc:`/installation/driver`
* :doc:`/installation/runtime`  
* Python 3.7 or later (refer to :any:`SetupPython` if necessary)
  

If the above requirements have been fulfilled, you can install the FuriosaAI NPU Python SDK library through PyPi.

.. code-block:: sh

  pip install --upgrade 'furiosa-sdk[runtime]~=0.1.0'


Running Basic Python Code
=================================

After the installation is complete, you can check whether the package is installed correctly by running the Python code as follows 
and outputting the version information.

.. code-block::
  
  >> from furiosa import runtime
  INFO:furiosa.runtime._api.v1:successfully loaded dynamic library libnux.so.1.0.0

  >> runtime.__full_version__
  'Furiosa SDK Runtime .release:0.1.1+907338a44e91f176495b3c24fce3d9b1e626a662 (libnux 0.3.0-dev 9418048e4 2021-03-29 02:59:26)'


Inference can be executed by loading the model and creating a ``session`` object.
When first loading the model, the process of compiling and optimizing the model is performed internally. 
This process is performed only when the model is initially loaded and may take from several seconds to 
tens of seconds depending on the model. (Very awk, no set time frame?*)

.. code-block::

  >>> sess = session.create('./MNISTnet_uint8_quant_without_softmax.tflite')
  num_slices: 16
  num_slices: 16
  [1/6] 🔍   Compiling from tflite to dfg
  Done in 0.001876285s
  [2/6] 🔍   Compiling from dfg to ldfg
  ▪▪▪▪▪ [1/3] Splitting graph...Done in 1.2222888s
  ▪▪▪▪▪ [2/3] Lowering...Done in 0.24661668s
  ▪▪▪▪▪ [3/3] Precalculating operators...Done in 0.007146129s
  Done in 1.4769053s
  [3/6] 🔍   Compiling from ldfg to cdfg
  Done in 0.000111313s
  [4/6] 🔍   Compiling from cdfg to gir
  Done in 0.004966648s
  [5/6] 🔍   Compiling from gir to lir
  Done in 0.000273256s
  [6/6] 🔍   Compiling from lir to enf
  Done in 0.003491739s
  ✨  Finished in 1.4908016s


After loading a model, you can print its information and see a list of input/output tensors and general information about each tensor.

.. code-block::

  >>> sess.print_summary()
  Inputs:
  {0: TensorDesc: shape=(1, 28, 28, 1), dtype=uint8, format=NHWC, size=784, len=784}
  Outputs:
  {0: TensorDesc: shape=(1, 1, 1, 10), dtype=uint8, format=NHWC, size=10, len=10}


You can run the inference by calling ``session.run()``. ``run()`` 
can take a list of `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
or `numpy.ndarray` as arguments.


For simple execution, let's use ``numpy`` to create a random tensor according to the input tensor 
and call ``run`` as follows.

.. code-block::

  >>> import numpy as np
  >>> input_meta = sess.inputs()[0]
  >>> input_meta
  TensorDesc: shape=(1, 28, 28, 1), dtype=uint8, format=NHWC, size=784, len=784
  
  >> input = np.random.randint(-128, 127, input_meta.shape(), dtype=np.int8)
  >>> outputs = sess.run(input)
  >>> outputs
  {0: <Tensor: shape=(1, 1, 1, 10), dtype=DataType.UINT8, numpy=[[[[255   0 239   0 183   0 209   0 255 255]]]]>}


The ``session.run(input)`` call uses the NPU to run the inference and returns a list of tensors. 
Since the returned result is a list, execute the following to get the `numpy.ndarray` of the first tensor.

.. code-block::

  print(outputs)
  print(outputs[0].numpy())


An example of the full code can be found at 
`furiosa-sdk-runtime/quickstart_example.py <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/furiosa-sdk-runtime/quickstart_example.py>`_.


See Also
=================================
* `Python SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/python/>`_
* `Python SDK Runtime examples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-runtime>`_
