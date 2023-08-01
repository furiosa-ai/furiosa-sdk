import pytest

from furiosa.runtime import is_legacy

if not is_legacy:
    pytest.skip("skipping legacy tests", allow_module_level=True)


def test_legacy_session():
    from furiosa.runtime.session import Session, create


def test_legacy_support():
    from furiosa.runtime.envs import (
        current_npu_device,
        is_compile_log_enabled,
        log_dir,
        profiler_output,
    )
