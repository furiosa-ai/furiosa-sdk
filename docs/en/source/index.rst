.. Furiosa SDK Documents master file, created by
   sphinx-quickstart on Tue Mar 23 11:18:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FuriosaAI SDK Documentation
=================================================

This document covers the process for installing and using the FuriosaAI SDK. 

.. note::

   The FuriosaAI SDK includes command line tools, runtime libraries, and Python libraries. 
   FuriosaAI NPU's kernel driver, firmware, and runtime are distributed according to 
   FuriosaAI's evaluation program registration and End User License Agreement. 
   For questions about downloading and evaluating the program, please contact us at contact@furiosa.ai.

SDK Installation 
-------------------------------------------------
* :doc:`SDK Installation Prerequisites (Required) </installation/prerequisites>`
* :doc:`NPU Kernel Driver and Firmware Installation (Required) </installation/driver>`
* :doc:`NPU Runtime Installation (Required)</installation/runtime>` : NPU Runtime and Application Tools
* :doc:`NPU Python SDK Installation (Optional)</installation/python-sdk>` : Python Library and Command Line Tool for NPU Usage
* :doc:`Web Service API key setting (Optional)</installation/apikey>` : API Key Setting for Using Tools Provided as Web Services

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
* :doc:`Command Line Tools Quickstart </quickstart/cli>`
* :doc:`Python SDK Quickstart </quickstart/python-sdk>`

Advanced Topics
-------------------
* :doc:`Model Quantization </advanced/quantization>`
* :doc:`NPU Acceleration Operators List </advanced/supported_operators>`
   

Code Examples
-------------------
* `Python SDK Quantization <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-sdk-quantizer>`_
* `Python SDK Inference <https://github.com/furiosa-ai/furiosa-sdk/tree/main/examples/furiosa-runtime>`_


Other links
-------------------
* `FuriosaAI Homepage <https://furiosa.ai>`_
* `FuriosaAI SDK Github <https://github.com/furiosa-ai/furiosa-sdk>`_
* `C Language SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/nux/>`_
* `Python SDK Reference <https://furiosa-ai.github.io/renegade-manual/references/python/>`_

