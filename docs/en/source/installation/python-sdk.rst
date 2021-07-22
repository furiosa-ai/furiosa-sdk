**********************************
FuriosaAI Python SDK Installation
**********************************

FuriosaAI's NPU Python SDK provides a number of interfaces for accelerating models using the NPU,
Python libraries, and command line tools.

Minimum Requirements
----------------------------------------
* :doc:`./driver`
* :doc:`./runtime`
* Python 3.7 or later (refer to :any:`SetupPython` if necessary)
* The latest version of pip (Upgrade to the latest version of pip with the following command)

  .. code-block::

        $ pip3 install --upgrade pip

Installation
----------------------------------------

The included Python library interface can be used with a number
of command line tools and functions. The FuriosaAI Python SDK can be installed with PyPi as follows.

.. code-block:: sh

  # Install the FuriosaAI NPU Python SDK to use the Python interface, e.g. `import furiosa`
  pip install --upgrade furiosa-sdk
  # Install additional tools, see below for details
  pip install --upgrade 'furiosa-sdk[runtime,quantizer]'
  # Install all additional tools
  pip install --upgrade 'furiosa-sdk[full]'

The following additional packages can be installed by using pip.

* ``cli``: Installs the command line tool. Refer to :doc:`/quickstart/cli` for usage

  .. code-block::

    pip install --upgrade 'furiosa-sdk[cli]'

* ``runtime``:  Installs various libraries to accelerating models using the NPU using the FuriosaAI NPU Runtime. **Required for model acceleration using the NPU**

  .. code-block::

    pip install --upgrade 'furiosa-sdk[runtime]'

* ``quantizer``: Installs the model quantization tool (see :doc:`/advanced/quantization`)

  .. code-block::

    pip install --upgrade 'furiosa-sdk[quantizer]'

* ``validator``: Installs model analysis tools, quantizes for accelerating corresponding models on the NPU, and includes compilation success verification tools.

  .. code-block::

    pip install --upgrade 'furiosa-sdk[quantizer,runtime,validator,cli]'


If you need a development environment for model inference and model quantization tools, the following installs them for you:

.. code-block:: sh

  pip install --upgrade 'furiosa-sdk[runtime,quantizer]'


Jupyter Notebook User Guide
----------------------------------------

While using Jupyter Notebook, you can leverage the FuriosaAI Python SDK
and various libraries in the Python ecosystem.

If you have already installed the Python SDK as above, you can just install
and use Jupyter Notebook using pip as shown below.
Because Jupyter Notebook installs a wide variety of dependencies,
:ref:`CondaInstall` is recommended.

.. code-block:: sh

  $ pip install jupyterlab
  $ jupyter-notebook
