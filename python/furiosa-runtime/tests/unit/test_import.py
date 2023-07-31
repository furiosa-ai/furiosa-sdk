import pytest


def test_utils():
    from furiosa.runtime import Axis, DataType, Model, ModelSource, Tensor, TensorDesc, is_legacy

    # FIXME: Cannot import TensorArray. See more details at
    # https://github.com/furiosa-ai/furiosa-sdk-private/issues/710
    with pytest.raises(ImportError):
        from furiosa.runtime import TensorArray


def test_runner():
    from furiosa.runtime.sync import Runner, Runtime, create_runner


def test_async_runner():
    from furiosa.runtime import Runner, Runtime, create_runner


def test_queue_api():
    from furiosa.runtime import Receiver, Submitter, create_queue


def test_sync_queue_api():
    from furiosa.runtime.sync import Receiver, Submitter, create_queue


def test_legacy_session():
    from furiosa.runtime.session import Session, create


def test_legacy_queue_api():
    from furiosa.runtime.session import AsyncSession, CompletionQueue, create_async


def test_profiler():
    from furiosa.runtime.profiler import RecordFormat, Resource, profile


def test_diagnostics():
    from furiosa.runtime import FuriosaRuntimeError, FuriosaRuntimeWarning, __full_version__


def test_legacy_support():
    from furiosa.runtime.envs import (
        current_npu_device,
        is_compile_log_enabled,
        log_dir,
        profiler_output,
    )
