**********************************************
Python SDK 빠르게 시작하기
**********************************************

FuriosaAI NPU Python SDK 라이브러리를 사용하면 NPU를 사용하는 프로그램을 Python으로 쉽게 작성할 수 있다.


**요구사항**

* :doc:`/installation/driver`
* :doc:`/installation/runtime`  
* Python 3.7 또는 상위 버전 (필요시 :any:`SetupPython` 참고)    
  

위 요구 사항이 준비된 경우 PyPi를 통해 FuriosaAI NPU Python SDK 라이브러리를 설치할 수 있다.

.. code-block:: sh

  pip install --upgrade furiosa-sdk[runtime]~=0.1.0


간단한 Python 코드 실행
=================================

설치가 완료된 후 다음과 같이 Python 코드를 실행하여 버전 정보를 출력하면 패키지가 잘 설치되었는지 확인해볼 수 있다.

.. code-block::
  
  >> from furiosa import runtime
  INFO:furiosa_sdk_runtime._api.v1:successfully loaded dynamic library libnux.so.1.0.0

  >> runtime.__full_version__
  'Furiosa SDK Runtime .release:0.1.1+907338a44e91f176495b3c24fce3d9b1e626a662 (libnux 0.3.0-dev 9418048e4 2021-03-29 02:59:26)'


모델을 로딩하여 ``session`` 객체를 생성하면 추론을 실행할 수 있다. 
모델을 처음 로딩하면 내부적으로 모델을 컴파일하고 최적화하는 과정이 수행된다.
이 과정은 모델을 최초 로딩할 때만 수행되고 모델에 따라 수 초에서 수십 초 가량이 소요된다.

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


모델을 로딩하고 난 뒤에는 로딩된 모델 정보를 출력해 볼 수 있다.
``Inputs``, ``Outputs`` 에서는 입력/출력 텐서 목록과 각 텐서에 대한 개괄적인 정보를 출력한다.

.. code-block::

  >>> sess.print_summary()
  Inputs:
  {0: TensorDesc: shape=(1, 28, 28, 1), dtype=uint8, format=NHWC, size=784, len=784}
  Outputs:
  {0: TensorDesc: shape=(1, 1, 1, 10), dtype=uint8, format=NHWC, size=10, len=10}


``session.run()`` 함수를 호출하여 추론을 실행할 수 있다. ``run()`` 는 
하나의 `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ 또는
`numpy.ndarray`_ 리스트를 인자로 받을 수 있다.

간단한 실행을 위해 ``numpy`` 를 이용해 입력 텐서에 맞게 랜덤 텐서를 아래와 같이 생성하여
``run`` 을 호출해보자.

.. code-block::

  >>> import numpy as np
  >>> input_meta = sess.inputs()[0]
  >>> input_meta
  TensorDesc: shape=(1, 28, 28, 1), dtype=uint8, format=NHWC, size=784, len=784
  
  >> input = np.random.randint(-128, 127, input_meta.shape(), dtype=np.int8)
  >>> outputs = sess.run(input)
  >>> outputs
  {0: <Tensor: shape=(1, 1, 1, 10), dtype=DataType.UINT8, numpy=[[[[255   0 239   0 183   0 209   0 255 255]]]]>}


``session.run(input)`` 호출은 NPU를 이용하여 추론을 실행하고 텐서 리스트를 반환한다.
반환된 결과가 리스트이므로 첫번째 텐서의 numpy.ndarray을 얻으려면 아래와 같이 실행한다.

.. code-block::

  print(outputs)
  print(outputs[0].numpy())


전체 코드를 담은 예제는 `furiosa-sdk-runtime/quickstart_example.py <https://github.com/furiosa-ai/furiosa-sdk/blob/main/examples/furiosa-sdk-runtime/quickstart_example.py>`_ 
에서 찾을 수 있다.


관련 문서
=================================
* `Python SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/python/>`_
* `Python SDK Runtime examples <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-runtime>`_
