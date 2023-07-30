furiosa.runtime package
=======================

.. module:: furiosa.runtime

.. role:: raw-html(raw)
    :format: html

.. default-role:: py:obj
.. highlight:: python

This package provides high-level Python APIs for Furiosa AI NPUs.




.. _`Runtime Variants`:

Runtime Variants
----------------

The package is divided into three main modules:

===========================   ==========   ============
Module                        Methods      Generation
===========================   ==========   ============
`furiosa.runtime`             Async        Current

`furiosa.runtime.sync`        Sync         Current

`furiosa.runtime.session`     Sync         :ref:`Legacy <Use of legacy modules>`
===========================   ==========   ============

The current generation API was first introduced in FuriosaRT 0.10.0.
The legacy generation API (`furiosa.runtime.session`) is still supported for the backward compatibility
but slated for removal in the future release.
See :ref:`Use of legacy modules` for the more information.

Each module further contains two different sets of interfaces:

.. glossary::

    Runners
        Runners provide a single class with ``run`` method.

        Multiple ``run`` calls can be active at any time,
        which is either possible by multiple tasks (for async modules) or threads (for sync modules).

    Queues
        Queues provide two separate classes with ``send`` and ``recv`` methods respectively.
        Each ``send`` should be paired with a context value to distinguish different ``recv`` outputs.

        While multiple inputs can be sent at any time,
        only a single task or thread should call ``recv``.


.. _`Use of legacy modules`:

Use of legacy modules
^^^^^^^^^^^^^^^^^^^^^

.. deprecated:: 0.10.0
    Any new use is strongly discouraged.

The package contains many historical modules,
including `furiosa.runtime.session` and `furiosa.runtime.errors`
(see :ref:`Legacy Supports` for the full list).
As of 0.10.0 they are deprecated and will report deprecation warnings.

Legacy modules are largely wrappers around current APIs by default,
as such there exist some slight incompatibilities,
most notably the lack of :ref:`error subclasses <module-furiosa.runtime.errors>`.
Those intercompatibilities are marked as **legacy only** or **current only**.

If this is not desirable you may enable the ``[legacy]`` extra on install
to force the old implementation and disable compatibility warnings.
Please note that you *cannot* use current-only APIs when the extra is enabled.
For example:

.. TODO: link to "install" (and make sure to explain extras)

.. |Legacy only| replace:: :ref:`Legacy only <Use of legacy modules>`
.. |legacy only| replace:: :ref:`legacy only <Use of legacy modules>`
.. |Current only| replace:: :ref:`Current only <Use of legacy modules>`
.. |current only| replace:: :ref:`current only <Use of legacy modules>`

=================   ===================================   =====================   =================
Availability        Example                               Default                 With ``[legacy]``
=================   ===================================   =====================   =================
Current only        `furiosa.runtime.Runtime`             |Available|             |Not available|

Current & Legacy    `furiosa.runtime.full_version`        |Available|             |Available|

Legacy              `furiosa.runtime.session.Session`     |Available but warns|   |Available|

Legacy only         `furiosa.runtime.errors.NativeError   |Not available|         |Available|
                    <furiosa.runtime.errors>`
=================   ===================================   =====================   =================

.. |Available| raw:: html

    <span style="color:green" title="Available">✔</span>

.. |Not available| raw:: html

    <span style="color:red" title="Not available">✘</span>

.. |Available but warns| raw:: html

    <span style="color:yellow" title="Available but warns">⚠️</span>



.. _`Model Inputs`:

Model Inputs
------------

.. currentmodule:: furiosa.runtime

.. class:: ModelSource

    Represents a value that specifies how to load the model.
    This is *not* a real class, but a type alias for either:

    1. Any `~pathlib.Path`-like type or a string, representing a path to the model file.

    2. Bytes, byte array or any class with ``__bytes__`` method, representing a raw model data.

    Future versions may allow additional types.

    .. versionadded:: 0.10

See :ref:`Compiler` for supported model formats and restrictions.





.. _`Tensor Inputs and Outputs`:

Tensor Inputs and Outputs
-------------------------

.. currentmodule:: furiosa.runtime

.. _Numpy: https://numpy.org/

Tensors use a `numpy.ndarray` type from Numpy_ as a primary representation,
but the following type aliases exist for documentation and typing purposes.

.. class:: TensorArray

    A type alias for a list of input tensors.
    This is *not* a real class, but can be either:

    * Any iterable of `Tensor`\s (but the iterable itself shouldn't be an `~numpy.ndarray`)
    * A single `Tensor`, if only one tensor is required

    The corresponding output type is always a list of `~numpy.ndarray`\s.

    .. versionadded:: 0.10

.. class:: Tensor

    A type alias for a single input tensor.
    This is *not* a real class, but can be either:

    * A single `numpy.ndarray`
    * A single scalar value or `numpy.generic`, if the tensor was zero-dimensional
    * Any other value that implements the NumPy `array interface`_

    .. _`array interface`: https://numpy.org/doc/stable/reference/arrays.interface.html

    The corresponding output type is always a single `~numpy.ndarray`.

    .. versionadded:: 0.10



.. _`Legacy Tensor Inputs and Outputs`:

Legacy Interface
^^^^^^^^^^^^^^^^^

.. currentmodule:: furiosa.runtime.tensor

Legacy interfaces support the same input and output types,
but due to the technical reasons, following concrete types are used for tensor outputs.

.. class:: TensorArray

    A subclass of `list`, so that all items should be `Tensor` and additional methods are provided.

    |Legacy only|:
    This class used to be not subclassed from `list` and
    supported only ``len(array)``, ``array[i]`` and ``array[i] = tensor``.

    .. method:: is_empty()

        :code:`True` if the array is empty.

        :rtype: bool

    .. method:: view()

        Returns a list of `~numpy.ndarray` views to all tensors, as in `Tensor.view`.

        :rtype: list[numpy.ndarray]

    .. method:: numpy()
        
        Returns a list of *copies* of `~numpy.ndarray` from all tensors, as in `Tensor.numpy`.

        :rtype: list[numpy.ndarray]

    .. _`Legacy TensorArray as input`:

    .. topic:: When used as an input type

        This type can be also internally converted from other types:

        * A list of :ref:`any types <Legacy Tensor as input>` that can be converted to `Tensor`
        * A single type that can be converted to `Tensor`, if only one tensor is required

.. class:: Tensor

    A subclass of `numpy.ndarray` providing additional methods.

    |Legacy only|:
    This class used to be not subclassed from `numpy.ndarray` and
    no methods were available except documented here.

    .. property:: shape
        :type: tuple[int, ...]

        Same as ``tensor.view().shape``.

    .. property:: numpy_dtype
        :type: type

        Same as ``tensor.view().dtype.type``.
        Returns a Numpy *type object* like `numpy.int8`.

        .. deprecated:: 0.10
            Contrary to its name, it didn't return `numpy.dtype` which was misleading and thus deprecated.
            Use `tensor.view().dtype <numpy.ndarray.dtype>` instead.

    .. method:: copy_from(data)

        Replaces the entire tensor with given data.

        :param data: What to replace this tensor
        :type data: numpy.ndarray or numpy.generic

    .. method:: view()

        Returns a `~numpy.ndarray` view to this tensor.
        Multiple views to the same tensor refer to the same memory region.

        :rtype: numpy.ndarray

    .. method:: numpy()

        Returns a *copy* of `~numpy.ndarray` from this tensor.
        Multiple copies to the same tensor are independent from each other and the original tensor.

        :rtype: numpy.ndarray

    .. _`Legacy Tensor as input`:

    .. topic:: When used as an input type

        This type can be also internally converted from other types:

        * A single `numpy.ndarray`
        * A single `numpy.generic`, if the tensor was zero-dimensional






.. _`Runtime Object`:

Runtime Object
--------------

.. currentmodule:: furiosa.runtime

Runtime objects are associated with a set of NPU devices and can be used to create inference sessions.

.. versionadded:: 0.10
    Previously sessions could only be created directly via
    `~furiosa.runtime.session.create` and `~furiosa.runtime.session.create_async` functions.

.. class:: Runtime(device=None)

    Asynchronous runtime object.

    :param device: A textual `device specification`_, see the section for defaults
    :type device: str or None

    This type is mainly used as a context manager::

        async with Runtime() as runtime:
            async with runtime.create_runner("path/to/model.onnx") as runner:
                outputs = await runner.run(inputs)

    The runtime acts as a scope for all subsequent inference sessions.
    Its lifetime starts when the object is successfully created,
    and ends when the `.close` method is called for the first time.
    All other methods will fail when the runtime has been closed.

    .. method:: close()
        :async:

        Tries to close the runtime if not yet closed.
        Waits until the runtime is indeed closed,
        or it takes too much time to close (in which case the runtime may still be open).

        The context manager internally calls this method at the end,
        and warns when the timeout has been reached.

        :return: :code:`True` if the runtime has been closed in time.

    Most other methods are documented in the :ref:`Runner API` and :ref:`Queue API` sections.

.. currentmodule:: furiosa.runtime.sync

.. class:: Runtime(device=None)

    Same as `furiosa.runtime.Runtime`, but all ``async`` methods are made synchronous.

    ::

        with Runtime() as runtime:
            with runtime.create_runner("path/to/model.onnx") as runner:
                outputs = runner.run(inputs)



.. _`Device Specification`:

Device Specification
^^^^^^^^^^^^^^^^^^^^

Runtime identifies a set of NPU devices by a textual string.

Implicit, any available device
    ``ARCH(X)*Y`` denotes any set of available devices where:

    * *ARCH* is a target architecture and currently should be ``warboy``.
    * *X* is the number of PEs to be used per each device.
    * *Y* is the number of devices to be used.

    ``(1)`` can be omitted, so for example ``warboy*1`` is a valid specification.

Device and PE index pair
    ``npu:X:Y`` denotes a PE number *Y* in the device number *X*,
    where *X* and *Y* are 0-based indices.

    ``npu:X:Y-Z`` denotes a fused PE made of two PEs ``npu:X:Y`` and ``npu:X:Z``.
    Intermediate tensors may occupy multiple PE worth of memory in this mode.

    .. TODO: properly document `npu:X` once it is fully supported and tested
    .. TODO: explain more about fusion and multi-core mode somewhere else

    .. note::
        Device and PE indices are determined by the kernel driver,
        and so should not be heavily relied upon especially when there are multiple devices.

Raw device name
    ``npuXpeY`` and ``npuXpeY-Z`` are same as ``npu:X:Y`` and ``npu:X:Y-Z`` respectively.

    Those names are same as raw device file names in ``/dev``.

    .. deprecated:: 0.10.0

Multiple devices
    Aforementioned specifications can be connected with ``,``.
    For example ``npu:1:0,npu:1:1`` is a valid specification.

    .. warning::
        Implicit specifications will allocate devices in greedy manner,
        so the runtime may fail to initialize even when there exist valid allocations.
        Mixed specifications like ``warboy*1,npu:1:0`` is not recommended for this reason.

Environment variables
    Following environment variables will be used for the default device specification
    when no explicit device specification is given.

    .. envvar:: FURIOSA_DEVICES

        .. versionadded:: 0.10

        Takes precedence over :envvar:`NPU_DEVNAME` if given.

        This environment variable is |current only|.

    .. envvar:: NPU_DEVNAME

        .. deprecated:: 0.10
            Use :envvar:`FURIOSA_DEVICES` instead.

    .. note::
        Environment variables can never override the explicit specification.

Default devices
    If there were no device specification or relevant environment variables,
    then it's assumed that there is a single Warboy device with 2 fused PEs and
    the default specification ``warboy(2)*1`` is used.





.. _`Model Metadata`:

Model Metadata
--------------

.. currentmodule:: furiosa.runtime

.. versionchanged:: 0.10
    Those types were provided by the `furiosa.runtime.model` module.

.. class:: Axis

    Represents an inferred axis of tensors.

    This type is used only for compiler diagnostics and doesn't affect the execution.

    .. property:: WIDTH
                  HEIGHT
                  CHANNEL
                  BATCH
                  UNKNOWN
        :classmethod:
        :type: Axis

        Constants for well-known axes:

        =========  ============  ============================================
        Property   Abbreviation  Description
        =========  ============  ============================================
        `WIDTH`    ``W``         Width

        `HEIGHT`   ``H``         Height

        `CHANNEL`  ``C``         Depth, or input channel for convolutions

        `BATCH`    ``N``         Batch, or output channel for convolutions

        `UNKNOWN`  ``?``         Other axes compiler has failed to infer
        =========  ============  ============================================

        This enumeration also contains some private axes internally used by the compiler.
        Their names and meanings are not stable and can change at any time.

.. class:: DataType(v)

    Represents a data type for each item in the tensor.

    :param v: A value to determine its data type
    :type v: `~numpy.dtype` or `~numpy.ndarray`

    The constructor can be also used to determine an NPU-supported data type from other objects::

        import numpy as np
        DataType(np.float32)                       # => DataType.FLOAT32
        DataType(np.zeros((3, 3), dtype='int8'))   # => DataType.INT8

    .. property:: FLOAT16
                  FLOAT32
                  BFLOAT16
                  INT8
                  INT16
                  INT32
                  INT64
                  UINT8
        :classmethod:
        :type: DataType

        Constants for supported types:

        ==========  =================================================================
        Property    Description
        ==========  =================================================================
        `FLOAT16`   IEEE 754 half-precision (binary16_) floating point type

        `FLOAT32`   IEEE 754 single-precision (binary32_) floating point type

        `BFLOAT16`  Bfloat16_ floating point type

        `INT8`      8-bit signed integer type

        `INT16`     16-bit signed integer type

        `INT32`     32-bit signed integer type

        `INT64`     64-bit signed integer type

        `UINT8`     8-bit unsigned integer type
        ==========  =================================================================

        .. _binary16: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
        .. _binary32: https://en.wikipedia.org/wiki/Double-precision_floating-point_format
        .. _Bfloat16: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

        This enumeration also contains some private types internally used by the compiler.
        Their representations are not stable and can change at any time.

    .. property:: numpy
        :type: numpy.dtype

        Returns a corresponding Numpy data type object.

        :raises ValueError: if this type has no Numpy equivalent

    .. property:: numpy_dtype
        :type: type

        Returns a Numpy *type object* corresponding to this data type, like `numpy.int8`.

        :raises ValueError: if this type has no Numpy equivalent

        .. deprecated:: 0.10
            Contrary to its name, it didn't return `numpy.dtype` which was misleading and thus deprecated.
            Use the `.numpy` property instead.

.. class:: TensorDesc

    Describes a single tensor in the input or output.

    .. property:: name
        :type: str or None

        A tensor name if any. This is only available for some model formats.

    .. property:: ndim
        :type: int

        The number of dimensions.

    .. method:: dim(idx)

        The size of `idx`-th dimension.

        :param int idx: 0-based dimension index
        :rtype: int

    .. property:: shape
        :type: tuple[int, ...]

        The shape of given tensor.

        :code:`desc.shape` is conceptually same as :code:`(desc.dim(0), ..., desc.dim(desc.ndim - 1))`.

    .. method:: axis(idx)

        The compiler-inferred axis for `idx`-th dimension. Defaults to `Axis.UNKNOWN`.

        :param int idx: 0-based dimension index
        :rtype: Axis

    .. property:: size
        :type: int

        The byte size of given tensor.

    .. method:: stride(idx)

        The stride of `idx`-th dimension in **items**.
        It's a distance between two adjacent elements in given dimension;
        this convention notably differs from `ndarray.strides` which is in bytes.

        :param int idx: 0-based dimension index
        :rtype: int

    .. property:: length
        :type: int

        The number of total elements in given tensor.

    .. property:: format
        :type: str

        A concatenation of abbreviated axes in given tensor, e.g. ``NCHW``.

    .. property:: dtype
        :type: DataType

        A data type for each element.

    .. property:: numpy_dtype
        :type: numpy.dtype

        A Numpy_ data type for each element. Same as `desc.dtype.numpy`.

.. class:: Model

    Describes a single model with possibly multiple input and output tensors.

    .. property:: input_num
        :type: int

        The number of input tensors.

    .. property:: output_num
        :type: int

        The number of output tensors.

    .. method:: input(idx)

        A description for `idx`-th input tensor.

        :param int idx: 0-based tensor index
        :rtype: TensorDesc

    .. method:: output(idx)

        A description for `idx`-th output tensor.

        :param int idx: 0-based tensor index
        :rtype: TensorDesc

    .. method:: inputs()

        A list of descriptions for each input tensor.

        :rtype: list[TensorDesc]

    .. method:: outputs()

        A list of descriptions for each output tensor.

        :rtype: list[TensorDesc]

    .. method:: summary()

        A human-readable summary of given model::

            >>> print(model.summary())
            Inputs:
            {0: TensorDesc(shape=(1, 1, 28, 28), dtype=FLOAT32, format=NCHW, size=3136, len=784)}
            Outputs:
            {0: TensorDesc(shape=(1, 10), dtype=FLOAT32, format=??, size=40, len=10)}

        :rtype: str

    .. method:: print_summary()

        Same as :code:`print(model.summary())`.





..
    Editor's note:

    Runner and queue sections have duplicate descriptions,
    which are intentional as it's hard to factor shared portions out of those sections
    without hampering readability.
    As such they should be synchronized to each other whenever changes are made.





.. _`Runner API`:

Runner API
----------

.. currentmodule:: furiosa.runtime

.. versionadded:: 0.10

Runner APIs provide a simple functional interface via a single `Runner` class.

.. method:: furiosa.runtime.Runtime.create_runner(model, *, worker_num=None, batch_size=None)
    :async:

    Creates a new inference session for given model.

    :param ModelSource model: Model path or data
    :param worker_num: The number of worker threads
    :type worker_num: int or None
    :param batch_size: The number of batches per each run
    :type batch_size: int or None
    :rtype: Runner

.. function:: create_runner(model, *, device=None, worker_num=None, batch_size=None)
    :async:

    Same as above, but the runtime is implicitly created and
    will be closed when the session couldn't be created or gets closed.

    See `~furiosa.runtime.Runtime.create_runner` and `Runtime` for arguments.

.. class:: Runner

    An inference session
    returned by `Runtime.create_runner <furiosa.runtime.Runtime.create_runner>` or `~furiosa.runtime.create_runner`.

    This type is mainly used as a context manager::

        async with runtime.create_runner("path/to/model.onnx") as runner:
            outputs = await runner.run(inputs)

    .. property:: model
        :type: ~furiosa.runtime.Model

        Informations about the associated model::

            async with runtime.create_runner("path/to/model.onnx") as runner:
                model = session.model
                for i in range(model.num_inputs):
                    print(f"Input tensor #{i}:", model.input(i))

        When the batch size is given,
        the first dimension of all input and output tensors is multiplied by that batch size.
        This dimension generally corresponds to the `~Axis.BATCH` axis.

    .. method:: run(inputs)
        :async:

        Runs a single inference.

        :param TensorArray inputs: Input tensors
        :rtype: list[numpy.ndarray]

        Input tensors are **not** copied to a new buffer.
        Modifications during the inference may result in an unexpected output;
        the runtime only ensures that such modifications do not cause a crash.

    .. method:: close()
        :async:

        Tries to close the session if not yet closed.
        Waits until the session is indeed closed,
        or it takes too much time to close (in which case the runtime may still be open).
        If the session was created via the top-level `~furiosa.runtime.create_runner` function,
        the implicitly initialized runtime is also closed as well.

        The context manager internally calls this method at the end,
        and warns when the timeout has been reached.
        It is also called at the unspecified point after `Runner` is subject to the garbage collection.

        :return: :code:`True` if the session (and the runtime if any) has been closed in time.

.. currentmodule:: furiosa.runtime.sync

Synchronous versions of those interfaces are also available through `furiosa.runtime.sync`:

.. method:: furiosa.runtime.sync.Runtime.create_runner(model, *, worker_num=None, batch_size=None)

    Synchronous version of `Runtime.create_runner <furiosa.runtime.Runtime.create_runner>` above.

.. function:: create_runner(model, *, worker_num=None, batch_size=None)

    Synchronous version of `~furiosa.runtime.create_runner` above.

.. class:: Runner

    Synchronous version of `~furiosa.runtime.Runner` above.



.. _`Legacy Runner API`:

Legacy Interface
^^^^^^^^^^^^^^^^

.. currentmodule:: furiosa.runtime.session

.. function:: create(model, *, device=None, worker_num=None, batch_size=None, compiler_hints=None)

    Compiles given model if needed, allocate an NPU device, and initializes a new inference session.

    :param ModelSource model: Model path or data
    :param device: A textual `device specification`_, see the section for defaults
    :type device: str or None
    :param worker_num: The number of worker threads
    :type worker_num: int or None
    :param batch_size: The number of batches per each run
    :type batch_size: int or None
    :param compiler_hints: If :code:`True`, compiler prints additional hint messages
    :type compiler_hints: bool or None
    :rtype: Session

    .. Editor's note: The following warning ensures that we don't have to document `compiler_config`

    .. versionchanged:: 0.10
        All optional arguments are now keyword-only.
        Positional arguments are still accepted and behave identically for now,
        but will warn against the future incompatibility.

.. class:: Session

    An inference session returned by `create`.

    This type is mainly used as a context manager::

        with create("path/to/model.onnx") as session:
            outputs = session.run(inputs)

    .. topic:: Model informations

        .. property:: model
            :type: ~furiosa.runtime.Model

            Informations about the associated model::

                with create("path/to/model.onnx") as session:
                    model = session.model
                    for i in range(model.num_inputs):
                        print(f"Input tensor #{i}:", model.input(i))

            .. versionadded:: 0.10

        For the compatibility, all `~furiosa.runtime.Model` properties and methods are
        also directly available to `Session`::

            with create("path/to/model.onnx") as session:
                for i in range(session.num_inputs):
                    print(f"Input tensor #{i}:", session.input(i))

        .. deprecated:: 0.10
            This mode of operation is no longer preferred, use `.model` instead.

        When the batch size is given,
        the first dimension of all input and output tensors is multiplied by that batch size.
        This dimension generally corresponds to the `~Axis.BATCH` axis.

    .. topic:: Inference

        .. method:: run(inputs)

            Runs a single inference.

            :param inputs: Input tensors
            :type inputs: :ref:`any TensorArray-like types <Legacy TensorArray as input>`
            :rtype: ~furiosa.runtime.tensor.TensorArray

            Unlike the current version,
            input tensors are copied to a new buffer unless it's a `~furiosa.runtime.tensor.TensorArray`,
            in which case it should not be altered after it was sent to `run`.

        .. method:: run_with(outputs, inputs)

            Runs a single inference, but all tensors are referred with explicit names.

            :param outputs: Output tensor names
            :type outputs: list[str]
            :param inputs: Input tensors and their names
            :type inputs: dict[str, ~numpy.ndarray]
            :returns: A list of output tensors, in the specified order
            :rtype: ~furiosa.runtime.tensor.TensorArray

            Input tensors are always copied to a new buffer.

            This method doesn't allow a partial input and all input tensors should be present.
            However there can be as few as a single output tensor specified.

    .. topic:: Miscellaneous

        .. method:: close()

            Tries to close the session if not yet closed.

            The context manager internally calls this method at the end.
            It is also called at the unspecified point after `Session` is subject to the garbage collection.





.. _`Queue API`:

Queue API
---------

.. currentmodule:: furiosa.runtime

.. versionadded:: 0.10

Queue APIs provide two objects `Submitter` and `Receiver` for separately handling inputs and outputs.
These are named so because they represent two queues around the actual processing:

.. glossary::

    Input queue
        Holds submitted input tensors until some worker is available and can process them.

    Output queue
        Holds output tensors that have been completed processed but not yet read.

Both have a configurable but finite size,
so submitting inputs too quickly or failing to receive outputs in time will block further processing.

.. method:: furiosa.runtime.Runtime.create_queue(model, *, worker_num=None, batch_size=None, input_queue_size=None, output_queue_size=None)
    :async:

    Creates a new inference session for given model.

    :param ModelSource model: Model path or data
    :param worker_num: The number of worker threads
    :type worker_num: int or None
    :param batch_size: The number of batches per each run
    :type batch_size: int or None
    :param input_queue_size: The :term:`input queue` size
    :type input_queue_size: int or None
    :param output_queue_size: The :term:`output queue` size
    :type output_queue_size: int or None
    :rtype: tuple[Submitter, Receiver], but see below

    It is also possible to call this function directly as an asynchronous context manager::

        async with runtime.create_queue("path/to/model.onnx",
                                        ) as (submitter, receiver):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(submit_task(submitter))
                tg.create_task(recv_task(receiver))

.. function:: create_queue(model, *, device=None, worker_num=None, batch_size=None, input_queue_size=None, output_queue_size=None)
    :async:

    Same as above, but the runtime is implicitly created and
    will be closed when the session couldn't be created or gets closed.

    See `~furiosa.runtime.Runtime.create_queue` and `Runtime` for arguments.

.. class:: Submitter

    A submitting half of the inference session
    returned by `Runtime.create_queue <furiosa.runtime.Runtime.create_queue>` or `~furiosa.runtime.create_queue`.

    This type is mainly used as a context manager::

        submitter, receiver = await runtime.create_queue("path/to/model.onnx")
        async with submitter:
            await submitter.submit(inputs)
        async with receiver:
            _, outputs = await receiver.recv()

    .. property:: model
        :type: ~furiosa.runtime.Model

        Informations about the associated model::

            submitter, receiver = await runtime.create_queue("path/to/model.onnx")
            async with submitter:
                model = submitter.model
                for i in range(model.num_inputs):
                    print(f"Input tensor #{i}:", model.input(i))

        When the batch size is given,
        the first dimension of all input and output tensors is multiplied by that batch size.
        This dimension generally corresponds to the `~Axis.BATCH` axis.

    .. method:: allocate()

        Returns a list of fresh tensors, suitable as input tensors.
        Their initial contents are not specified (but probably zeroed).

        :rtype: list[numpy.ndarray]

        While this is no different than creating tensors yourself,
        the runtime may allocate tensors in the device-friendly way
        so it is recommended to use this method whenever appropriate.

    .. method:: submit(inputs, context=None)
        :async:

        Submits a single inference with an associated value.

        :param TensorArray inputs: Input tensors
        :param context: Associated value, to distinguish output tensors from `Receiver`

        The method will return immediately,
        unless the :term:`input queue` is full in which case the method will be blocked.
        Output tensors would be available through `Receiver` later.

        Input tensors are **not** copied to a new buffer.
        Modifications during the inference may result in an unexpected output;
        the runtime only ensures that such modifications do not cause a crash.

        .. warning::

            Associated values should be simple values like an integer or a `UUID <uuid.uuid4>`,
            as they are retained as long as a progress can be made and can cause logical memory leaks.

    .. method:: close()
        :async:

        Tries to close the submitter if not yet closed.
        Waits until the submitter is indeed closed, or it takes too much time to close.

        This does *not* close the corresponding `Receiver`.
        Any remaining tensors in the :term:`input queue` will be processed nevertheless,
        and then the :term:`output queue` will be signaled that no more results would be returned.

        The context manager internally calls this method at the end,
        and warns when the timeout has been reached.
        It is also called at the unspecified point after `Submitter` is subject to the garbage collection.

        :return: :code:`True` if the submitter has been closed in time.

.. class:: Receiver

    A receiving half of the inference session
    returned by `Runtime.create_queue <furiosa.runtime.Runtime.create_queue>` or `~furiosa.runtime.create_queue`.

    This type is mainly used as a context manager::

        submitter, receiver = await runtime.create_queue("path/to/model.onnx")
        async with submitter:
            await submitter.submit(inputs)
        async with receiver:
            _, outputs = await receiver.recv()

    .. property:: model
        :type: ~furiosa.runtime.Model

        Informations about the associated model::

            submitter, receiver = await runtime.create_queue("path/to/model.onnx")
            async with receiver:
                model = receiver.model
                for i in range(model.num_outputs):
                    print(f"Output tensor #{i}:", model.output(i))

        The same remarks to `Submitter` apply when the batch size is given.

    The type can be used as an asynchronous iterable,
    which only finishes when either the corresponding `Submitter` or the receiver itself has been closed::

        submitter, receiver = await runtime.create_async("path/to/model.onnx")
        task = asyncio.create_task(submit_task(submitter))
        async for context, outputs in receiver:
            handle_output(context, outputs)

    Note that in this usage a :code:`async with` block is not strictly needed
    as `Submitter` should have been already closed when the loop finishes.

    It is also possible to manually receive results:

    .. method:: recv()
        :async:

        Waits for a single inference result, and returns it with the associated value.

        :returns: A tuple of the associated value and output tensors, in this order
        :rtype: tuple[any, ~furiosa.runtime.tensor.TensorArray]

        The runtime guarantees that each inference result is received at most once,
        but the completion order may differ from the submission order.
        Put the associated value to `Submitter.submit` to recover the original order.

        Multiple parallel `recv` calls are fine but do not have additional benefits.
        On the other hands, if no `recv` calls are made for a while,
        the :term:`output queue` eventually fills up and would block any further processing.

        .. note::
            This method does not support ``timeout`` unlike others,
            because `asyncio.timeout` provides an idiomatic way to do that::

                try:
                    async with asyncio.timeout(10):
                        context, outputs = await receiver.recv()
                except asyncio.TimeoutError:  # Not the built-in `TimeoutError`!
                    print('Timed out!')

    .. method:: close()
        :async:

        Tries to close the receiver if not yet closed.
        This also notifies the corresponding `Submitter` and will block further submissions.
        Waits until the receiver is indeed closed, or it takes too much time to close.

        If the receiver was created via the top-level `~furiosa.runtime.create_queue` function,
        the implicitly initialized runtime is also closed as well.

        The context manager internally calls this method at the end,
        and warns when the timeout has been reached.
        It is also called at the unspecified point after `Receiver` is subject to the garbage collection.

        :return: :code:`True` if the receiver (and the runtime if any) has been closed in time.

.. currentmodule:: furiosa.runtime.sync

Synchronous versions of those interfaces are also available through `furiosa.runtime.sync`:

.. method:: furiosa.runtime.sync.Runtime.create_queue(model, *, device=None, worker_num=None, batch_size=None, input_queue_size=None, output_queue_size=None)

    Synchronous version of `Runtime.create_queue <furiosa.runtime.Runtime.create_queue>` above.

.. function:: create_queue(model, *, device=None, worker_num=None, batch_size=None, input_queue_size=None, output_queue_size=None)

    Synchronous version of `~furiosa.runtime.create_queue` above.

.. class:: Submitter

    Synchronous version of `~furiosa.runtime.Submitter` above.

.. class:: Receiver

    Synchronous version of `~furiosa.runtime.Receiver` above, with the following exception:

    .. method:: recv(timeout=None)

        Unlike an asynchronous version,
        this method *does* accept the timeout because it is otherwise impossible to specify the timeout in a synchronous context.

        :param timeout: The timeout in *seconds*
        :type timeout: float or None
        :raises TimeoutError: When ``timeout`` is given and the timeout has been reached

        .. note::
            Unlike `CompletionQueue.recv`, this method raises a standard Python exception.



.. _`Legacy Queue API`:

Legacy Interface
^^^^^^^^^^^^^^^^

.. currentmodule:: furiosa.runtime.session

.. function:: create_async(model, *, device=None, worker_num=None, batch_size=None, compiler_hints=None, input_queue_size=None, output_queue_size=None)

    Compiles given model if needed, allocate an NPU device, and initializes a new inference session.

    :param ModelSource model: Model path or data
    :param device: A textual `device specification`_, see the section for defaults
    :type device: str or None
    :param worker_num: The number of worker threads
    :type worker_num: int or None
    :param batch_size: The number of batches per each run
    :type batch_size: int or None
    :param compiler_hints: If :code:`True`, compiler prints additional hint messages
    :type compiler_hints: bool or None
    :param input_queue_size: The :term:`input queue` size
    :type input_queue_size: int or None
    :param output_queue_size: The :term:`output queue` size
    :type output_queue_size: int or None
    :rtype: tuple[AsyncSession, CompletionQueue]

    .. Editor's note: The following warning ensures that we don't have to document `compiler_config`

    .. versionchanged:: 0.10
        All optional arguments are now keyword-only.
        Positional arguments are still accepted and behave identically for now,
        but will warn against the future incompatibility.

    |Legacy only|:
    The input and output queue were unbounded when their sizes are not given.
    This was never documented and the current API doesn't support unbounded queues.
    In order to faciliate migration, the legacy API will continue to use
    much larger default queue sizes though.

.. class:: AsyncSession

    A submitting half of the inference session returned by `create_async`.

    This type is mainly used as a context manager::

        session, queue = create_async("path/to/model.onnx")
        with session:
            session.send(inputs)
        with queue:
            _, outputs = queue.recv()

    .. topic:: Model informations

        .. property:: model
            :type: ~furiosa.runtime.Model

            Informations about the associated model::

                session, queue = create_async("path/to/model.onnx")
                with session:
                    model = session.model
                    for i in range(model.num_inputs):
                        print(f"Input tensor #{i}:", model.input(i))

            .. versionadded:: 0.10

        For the compatibility, all `~furiosa.runtime.Model` properties and methods are
        also directly available to `AsyncSession`::

            session, queue = create_async("path/to/model.onnx")
            with session:
                for i in range(session.num_inputs):
                    print(f"Input tensor #{i}:", session.input(i))

        .. deprecated:: 0.10
            This mode of operation is no longer preferred, use `.model` instead.

        When the batch size is given,
        the first dimension of all input and output tensors is multiplied by that batch size.
        This dimension generally corresponds to the `~Axis.BATCH` axis.

    .. topic:: Submitting

        .. method:: submit(inputs, context=None)

            Submits a single inference with an associated value.

            :param inputs: Input tensors
            :type inputs: :ref:`any TensorArray-like types <Legacy TensorArray as input>`
            :param context: Associated value, to distinguish output tensors from `CompletionQueue`

            The method will return immediately,
            unless the :term:`input queue` is full in which case the method will be blocked.
            Output tensors would be available through `CompletionQueue` later.

            Unlike the current version,
            input tensors are copied to a new buffer unless it's a `~furiosa.runtime.tensor.TensorArray`,
            in which case it should not be altered after it was sent to `submit`.

            .. warning::

                Associated values should be simple values like an integer or a `UUID <uuid.uuid4>`,
                as they are retained as long as a progress can be made and cause logical memory leaks.

    .. topic:: Miscellaneous

        .. method:: close()

            Tries to close the session if not yet closed.

            This does *not* close the corresponding `CompletionQueue`.
            Any remaining tensors in the :term:`input queue` will be processed nevertheless,
            and then the :term:`output queue` will be signaled that no more results would be returned.

            The context manager internally calls this method at the end.
            It is also called at the unspecified point after `AsyncSession` is subject to the garbage collection.

.. class:: CompletionQueue

    A receiving half of the inference session returned by `create_async`.

    This type is mainly used as a context manager::

        session, queue = create_async("path/to/model.onnx")
        with session:
            session.send(inputs)
        with queue:
            _, outputs = queue.recv()

    .. topic:: Model informations

        .. property:: model
            :type: ~furiosa.runtime.Model

            Informations about the associated model::

                session, queue = create_async("path/to/model.onnx")
                with queue:
                    model = queue.model
                    for i in range(model.num_outputs):
                        print(f"Output tensor #{i}:", model.output(i))

            .. versionadded:: 0.10

        For the compatibility, all `~furiosa.runtime.Model` properties and methods are
        also directly available to `CompletionQueue`::

            session, queue = create_async("path/to/model.onnx")
            with queue:
                for i in range(queue.num_outputs):
                    print(f"Output tensor #{i}:", queue.output(i))

        .. deprecated:: 0.10
            This mode of operation is no longer preferred, use `.model` instead.

        The same remarks to `AsyncSession` apply when the batch size is given.

    .. topic:: Receiving

        The type can be used as an iterable,
        which only finishes when either the corresponding `AsyncSession` or the queue itself has been closed::

            session, queue = create_async("path/to/model.onnx")
            spawn_thread_to_send_inputs(session)
            for context, outputs in queue:
                handle_output(context, outputs)

        Note that in this usage a :code:`with` block is not strictly needed
        as `AsyncSession` should have been already closed when the loop finishes.

        It is also possible to manually receive results:

        .. method:: recv(timeout=None)

            Waits for a single inference result, and returns it with the associated value.

            :param timeout: The timeout in *milliseconds*
            :type timeout: int or None
            :returns: A tuple of the associated value and output tensors, in this order
            :rtype: tuple[any, ~furiosa.runtime.tensor.TensorArray]
            :raises QueueWaitTimeout: When ``timeout`` is given and the timeout has been reached

            The runtime guarantees that each inference result is received at most once,
            but the completion order may differ from the submission order.
            Put the associated value to `AsyncSession.submit` to recover the original order.

            Multiple parallel `recv` calls are fine but do not have additional benefits.
            On the other hands, if no `recv` calls are made for a while,
            the :term:`output queue` eventually fills up and would block any further processing.

            .. versionchanged:: 0.10
                The millisecond convention for ``timeout`` is not Pythonic and prone to error.
                `~furiosa.runtime.Receiver` uses seconds instead,
                so this method will now warn for the potential future incompatibillity.

    .. topic:: Miscellaneous

        .. method:: close()

            Tries to close the completion queue if not yet closed.
            This also notifies the corresponding `AsyncSession` and will block further submissions.

            The context manager internally calls this method at the end.
            It is also called at the unspecified point after `CompletionQueue` is subject to the garbage collection.





.. _Profiler:

Profiler
--------

.. module:: furiosa.runtime.profiler

.. versionadded:: 0.10

The `furiosa.runtime.profiler` module provides a basic profiler facility.
This module requires Pydantic_.

.. _Pydantic: https://pydantic.dev

.. class:: RecordFormat

    Profiler format to record profile data.

    .. property:: ChromeTrace
                  PandasDataFrame
        :classmethod:
        :type: RecordFormat

.. class:: Resource

    Profiler target resource to be recorded.

    .. property:: ALL
                  CPU
                  NPU
        :classmethod:
        :type: Resource

.. class:: profile(resource=Resource.ALL, format=RecordFormat.ChromeTrace, **config)

    Profile context manager::

        from furiosa.runtime.profiler import RecordFormat
        with open("profile.json", "w") as f:
            with profile(format=RecordFormat.ChromeTrace, file=f) as profiler:
                # Profiler enabled from here
                with profiler.record("Inference"):
                    ... # Profiler recorded with span named 'Inference'

    :param Resource resource: Target resource to be profiled. e.g. CPU or NPU.
    :param RecordFormat format: Profiler format. e.g. ChromeTrace.
    :param config: Format specific config. You need to pass valid arguments for the format.
    :raise pydantic.error_wrappers.ValidationError: Raise when config validation failed.

    .. admonition:: Configuration arguments

        .. TODO: explain them

        `RecordFormat.ChromeTrace` supports:

        * **extra** (`str`) -- Defaults to :code:`"forbid"`
        * **arbitrary_types_allowed** (`bool`) -- Defaults to :code:`True`
        * **json_encoders** (`dict[type, callable]`) -- Defaults to a single callable for `io.IOBase`

        `RecordFormat.PandasDataFrame` supports:

        * **extra** (`str`) -- Defaults to :code:`"allow"`
        * **arbitrary_types_allowed** (`bool`) -- Defaults to :code:`True`

    .. method:: record(name='', warm_up=False)

        Create profiler span with specified name.

        :param str name: Profiler record span name.
        :param bool warm_up: If true, do not record profiler result, and just warm up.

    .. TODO: following methods are undocumented

    .. method:: get_pandas_dataframe()
    .. method:: get_pandas_dataframe_with_filter(column, value)
    .. method:: get_cpu_pandas_dataframe()
    .. method:: get_npu_pandas_dataframe()
    .. method:: print_npu_operators()
    .. method:: print_npu_executions()
    .. method:: print_external_operators()
    .. method:: print_inferences()
    .. method:: print_summary()
    .. method:: export_chrome_trace(filename)





.. _Diagnostics:

Diagnostics
-----------

.. currentmodule:: furiosa.runtime

.. function:: full_version()

    Returns a string for the full version of this package.

.. exception:: FuriosaRuntimeError

    A base class for all runtime exceptions.

    .. versionchanged:: 0.10
        Previously this was named ``NativeException``
        and was a subclass of `~furiosa.common.error.FuriosaError`.
        :ref:`A large number of subclasses <module-furiosa.runtime.errors>` were also removed,
        to make a room for the upcoming restructuring.

.. exception:: FuriosaRuntimeWarning

    A base class for all runtime warnings.

    .. versionadded:: 0.10

This package currently doesn't have a dedicated logging interface,
but the following environment variable can be used to set the basic logging level:

.. envvar:: FURIOSA_LOG_LEVEL

    If set, should be either one of the following, in the order of decreasing verbosity:

    * ``INFO`` (default)
    * ``WARN``
    * ``ERROR``
    * ``OFF``





.. _`Legacy Supports`:

Legacy Supports
---------------

The following submodules only exist to support :ref:`legacy codes <Runtime Variants>`.

.. deprecated:: 0.10
    All legacy submodules will warn on import and any major incompatibilites will be reported.
    Also unless otherwise stated, all functions and types described here are
    either not available or will always fail.
    The ``[legacy]`` extra may be used to disable these behavior
    at the expense of the lack of the current APIs.



furiosa.runtime.compiler
^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.compiler

.. function:: generate_compiler_log_path()

    Generates a log path for compilation log.

    :rtype: Path

    This function is |legacy only|.



furiosa.runtime.consts
^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.consts

This module is empty.



furiosa.runtime.envs
^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.envs

.. function:: current_npu_device()

    Returns the current npu device name.

    :returns: NPU device name
    :rtype: str

    This function is |legacy only|.

.. function:: is_compile_log_enabled()

    Returns True or False whether the compile log is enabled or not.

    :returns: True if the compile log is enabled, or False.
    :rtype: bool

    This function is |legacy only|.

.. function:: log_dir()

    Returns FURIOSA_LOG_DIR where the logs are stored.

    :returns: The log directory of Furiosa SDK
    :rtype: str

    This function is |legacy only|.

.. function:: profiler_output()

    Returns FURIOSA_PROFILER_OUTPUT_PATH where profiler outputs written.

    For compatibility, NUX_PROFILER_PATH is also currently being supported,
    but it will be deprecated by FURIOSA_PROFILER_OUTPUT_PATH later.

    :returns: The file path of profiler output if specified, or None.
    :rtype: str or None

    This function is |legacy only|.



.. _module-furiosa.runtime.errors:

furiosa.runtime.errors
^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.errors

.. exception:: IncompatibleModel
               CompilationFailed  
               InternalError
               UnsupportedTensorType
               UnsupportedDataType
               IncompatibleApiClientError
               InvalidYamlException
               ApiClientInitFailed
               NoApiKeyException
               InvalidSessionOption
               QueueWaitTimeout
               SessionTerminated
               DeviceBusy
               InvalidInput
               TensorNameNotFound
               UnsupportedFeature
               InvalidCompilerConfig
               SessionClosed

    Specific subclasses of `FuriosaRuntimeError`.

    They are |legacy only| and
    have been mostly replaced with standard Python exceptions like `TypeError` or `ValueError`,
    except for the following:

    * `DeviceBusy`
    * `InvalidInput`
    * `QueueWaitTimeout`
    * `SessionClosed`
    * `SessionTerminated`

This module also reexports `FuriosaRuntimeError` as ``NativeException``.



furiosa.runtime.model
^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.model

This module reexports `Model`.



furiosa.runtime.session
^^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.session

This module reexports `Session`, `AsyncSession`, `CompletionQueue`, `create` and `create_async`.



furiosa.runtime.tensor
^^^^^^^^^^^^^^^^^^^^^^

.. module:: furiosa.runtime.tensor

.. function:: numpy_dtype(value)

    Returns numpy dtype from any eligible object.

.. function:: zeros(desc)

    Returns a zero tensor matching given tensor description.

    :param TensorDesc desc: Tensor description

.. function:: rand(desc)

    Returns a random tensor matching given tensor description.

    :param TensorDesc desc: Tensor description

    This is meant to be a quick test function and
    no guarantees are made for quality, performance and correctness.

    |Legacy only|: The function was only correctly defined for floating point types.

This module also contains `~furiosa.runtime.tensor.Tensor` and `~furiosa.runtime.tensor.TensorArray`
which are described separately.

This module also reexports `Axis`, `DataType` and `TensorDesc`.
