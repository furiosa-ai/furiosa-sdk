.. _BugReport:

**********************************
버그 신고
**********************************

사용 중 해결되지 않는 문제를 겪을 경우 `FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_ 에서
버그 신고를 할 수 있다. 버그 신고에는 기본적으로 다음 정보가 포함되어야 한다.

#. 문제 재현 방법
#. 문제 발생 당시 로그 또는 스크린샷
#. SDK 버전 정보
#. 모델 컴파일 실패 시 컴파일러 로그

기본적으로 furiosa-sdk는 에러가 발생하는 경우 아래와 같은 정보를 출력한다.
아래와 같은 화면을 본다면, ``Information Dump`` 이하의 정보와 메시지에 출력되는 컴파일러 로그파일 (아래 예에서는 ``/home/furiosa/.local/state/furiosa/logs/compile-20211121223028-l5w4g6.log``)
을 `FuriosaAI 고객지원 센터 <https://furiosa-ai.atlassian.net/servicedesk/customer/portals>`_ 의 `Bug Report` 섹션에 신고하라.

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


위와 같은 정보가 출력되지 않는 경우 다음 방법으로 필요 정보를 직접 수집하여 `FuriosaAI 고객지원 센터`_ 에 버그 신고를 할 수 있다.

파이썬 런타임의 버전 정보는 다음과 같이 파악할 수 있다.

.. code-block::

    $ python --version
    Python 3.8.6

SDK 버전 정보는 다음과 같이 파악할 수 있다.

.. code-block::

    $ python -c "from furiosa import runtime;print(runtime.__full_version__)"
    loaded native library /usr/lib64/libnux.so (0.5.0 407c0c51f)
    Furiosa SDK Runtime 0.5.0 (libnux 0.5.0 407c0c51f 2021-11-18 22:32:34)
