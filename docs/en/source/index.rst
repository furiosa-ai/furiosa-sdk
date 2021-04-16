.. Furiosa SDK Documents master file, created by
   sphinx-quickstart on Tue Mar 23 11:18:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FuriosaAI SDK Documentation
=================================================

This Documentation covers the process for installing and using the FuriosaAI SDK. 

.. note::

   The FuriosaAI SDK includes command line tools, runtime libraries, and Python libraries. 
   FuriosaAI NPU's kernel driver, firmware, and runtime are distributed according to 
   FuriosaAI's evaluation program registration and End User License Agreement. 
   For questions about downloading and evaluating the program, please contact us at contact@furiosa.ai.

SDK installation 
-------------------------------------------------
* :doc:`SDK installation prerequisites (required) </installation/prerequisites>`
* :doc:`NPU kernel driver and firmware installation (required) </installation/driver>` : NPU runtime and NPU application tools
* :doc:`NPU Runtime Installation (required)</installation/runtime>` : NPU runtime and application tools
* :doc:`NPU Python SDK Installation (optional)</installation/python-sdk>` : Python library and command line tool for NPU usage
* :doc:`Web Service API key setting (optional)</installation/apikey>` : API key setting for using tools provided as web services

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: SDK installation

   /installation/prerequisites
   /installation/driver
   /installation/runtime
   /installation/python-sdk
   /installation/apikey

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Quickstart

   /quickstart/python-sdk
   /quickstart/cli

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Topics

   /advanced/quantization
   /advanced/supported_operators

Quickstart
-------------------
* :doc:`Command line tools Quickstart </quickstart/cli>`
* :doc:`Python SDK Quickstart </quickstart/python-sdk>`

Advanced Topics
-------------------
* :doc:`Model Quantization </advanced/quantization>`
* :doc:`NPU acceleration operators list </advanced/supported_operators>`
   

Code Examples
-------------------
* `Python SDK Quantization <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-quantizer>`_
* `Python SDK Inference <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-runtime>`_


Other links
-------------------
* `FuriosaAI Homepage <https://furiosa.ai>`_
* `FuriosaAI SDK Github <https://github.com/furiosa-ai/furiosa-sdk>`_
* `C Language SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/nux/>`_
* `Python SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/python/>`_

