Runtime
=======

Errors
------

Nux Exception and Error

.. automodule:: furiosa.runtime.errors
   :members: is_ok, is_err

NativeError
^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.NativeError
   :members: 

NuxException
^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.NuxException
   :members: 

IncompatibleModel
^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.IncompatibleModel
   :members: 

CompilationFailed
^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.CompilationFailed
   :members: 

InternalError
^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.InternalError
   :members: 

UnsupportedTensorType
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.UnsupportedTensorType
   :members: 

IncompatibleApiClientError
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.IncompatibleApiClientError
   :members: 

InvalidYamlException
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.InvalidYamlException
   :members: 

ApiClientInitFailed
^^^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.ApiClientInitFailed
   :members: 

NoApiKeyException
^^^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.errors.NoApiKeyException
   :members: 


Model
-----

Model and its methods to access model metadata

.. autoclass:: furiosa.runtime.model.Model
   :members: 

Session
-------

Session and its asynchronous API for model inference

.. automodule:: furiosa.runtime.session
   :members: create, create_async

Session
^^^^^^^

.. autoclass:: furiosa.runtime.session.Session
   :members: 

CompletionQueue
^^^^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.session.CompletionQueue
   :members: 

AsyncSession
^^^^^^^^^^^^

.. autoclass:: furiosa.runtime.session.AsyncSession
   :members: 

Tensor
------

Tensor object and its utilities

.. automodule:: furiosa.runtime.tensor
   :members: numpy_dtype
   :undoc-members:

Axis
^^^^

.. autoclass:: furiosa.runtime.tensor.Axis
   :members:

DataType
^^^^^^^^

.. autoclass:: furiosa.runtime.tensor.DataType
   :members:

TensorArray
^^^^^^^^^^^

.. autoclass:: furiosa.runtime.tensor.TensorDesc
   :members:

Tensor
^^^^^^

.. autoclass:: furiosa.runtime.tensor.Tensor
   :members:   

TensorArray
^^^^^^^^^^^

.. autoclass:: furiosa.runtime.tensor.TensorArray
   :members:

