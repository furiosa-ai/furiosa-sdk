.. _BugReport:

**********************************
Bug Report
**********************************

If you encounter an unresolvable issue, you can file a bug report at `FuriosaAI customer service center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_. 
The following information should be included in a bug report.

#. How to reproduce the bug  
#. Log or screenshot of the bug  
#. SDK version information 
#. Compilation log, if model compilation failed 

By default, when an error happens furiosa-sdk outputs the following message. 
If you see the following message, file a report with 
1) the information given below the ``Information Dump``, and 
2) the compilation log file (this would be ``/home/furiosa/.local/state/furiosa/logs/compile-20211121223028-l5w4g6.log``in the following example) outputted in the message,
in the `Bug Report` section of `FuriosaAI customer service center <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_

.. code-block::

    Saving the compilation log into /home/furiosa/.local/state/furiosa/logs/compile-20211121223028-l5w4g6.log
    Using furiosa-compiler 0.5.0 (rev: 407c0c51f-modified built at 2021-11-18 22:32:34)
    2021-11-22T06:30:28.392114Z  INFO Npu (npu0pe0) is being initialized
    2021-11-22T06:30:28.397757Z  INFO NuxInner create with pes: [PeId(0)]
    [1/6] 🔍   Compiling from onnx to dfg
    2021-11-22T06:30:28.423026Z  INFO [Profiler] Received a termination signal.
    2021-11-22T06:30:28.423371Z ERROR fail to compile the model: the static shape of tensor 'input' contains an unsupported dimension value: Some(DimParam("batch_size"))
    ================================================================================
    Information Dump
    ================================================================================
    - Python version: 3.8.10 (default, Sep 28 2021, 16:10:42)  [GCC 9.3.0]
    - furiosa-libnux path: libnux.so.0.5.0
    - furiosa-libnux version: 0.5.0 (rev: 407c0c51f built at 2021-11-18 22:32:34)
    - furiosa-compiler version: 0.5.0 (rev: 407c0c51f built at 2021-11-18 22:32:34)
    - furiosa-sdk-runtime version: Furiosa SDK Runtime  (libnux 0.5.0 407c0c51f 2021-11-18 22:32:34)

    Please check the compiler log at /home/furiosa/.local/state/furiosa/logs/compile-20211121223028-l5w4g6.log.
    If you have a problem, please report the log file to https://furiosa-ai.atlassian.net/servicedesk/customer/portals with the information dumped above.
    ================================================================================


If you do not see a message as shown above, refer to the instructions below to collect the necessary information yourself 
to file a bug report at `FuriosaAI customer service center`_

You can find the Python runtime version information as shown. 

.. code-block::

    $ python --version
    Python 3.8.6

You can find the SDK version information as shown. 

.. code-block::

    $ python -c "from furiosa import runtime;print(runtime.__full_version__)"
    loaded native library /usr/lib64/libnux.so (0.5.0 407c0c51f)
    Furiosa SDK Runtime 0.5.0 (libnux 0.5.0 407c0c51f 2021-11-18 22:32:34)
