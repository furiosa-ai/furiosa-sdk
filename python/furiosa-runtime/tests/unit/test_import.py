import pytest

from furiosa.runtime import is_legacy

if is_legacy:
    pytest.skip("skipping furiosa-rt tests", allow_module_level=True)


def test_utils():
    from furiosa.runtime import Axis, DataType, Model, ModelSource, Tensor, TensorArray, TensorDesc


def test_runner():
    from furiosa.runtime.sync import Runner, Runtime, create_runner


def test_async_runner():
    from furiosa.runtime import Runner, Runtime, create_runner


def test_queue_api():
    from furiosa.runtime import Receiver, Submitter, create_queue


def test_sync_queue_api():
    from furiosa.runtime.sync import Receiver, Submitter, create_queue


def test_profiler():
    from furiosa.runtime.profiler import RecordFormat, Resource, profile


def test_diagnostics():
    from furiosa.runtime import FuriosaRuntimeError, FuriosaRuntimeWarning, __full_version__


def test_legacy_session_and_queue_api():
    with pytest.warns(FutureWarning):
        from furiosa.runtime.session import (
            AsyncSession,
            CompletionQueue,
            Session,
            create,
            create_async,
        )


def test_legacy_support():
    with pytest.warns(FutureWarning):
        from furiosa.runtime.envs import (
            current_npu_device,
            is_compile_log_enabled,
            log_dir,
            profiler_output,
        )
