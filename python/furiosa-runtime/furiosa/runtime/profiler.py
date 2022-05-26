from __future__ import annotations

from contextlib import contextmanager
from ctypes import byref, c_void_p
from enum import Flag, IntEnum, auto
import io
import sys
from types import TracebackType
from typing import Any, Type

from pydantic import BaseModel, Field

from ._api import LIBNUX
from .errors import into_exception, is_err


class Resource(Flag):
    """Profiler target resource to be recorded."""

    CPU = auto()
    NPU = auto()
    ALL = CPU | NPU


class RecordFormat(IntEnum):
    """Profiler format to record profile data."""

    ChromeTrace = 0


class ChromeTraceConfig(BaseModel):
    """ChromeTrace specific config.

    Attributes:
        file: file descriptor to write profile data. By default, sys.stdout.
    """

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

        # Custom encoder for File like type.
        # See https://pydantic-docs.helpmanual.io/usage/exporting_models/#json_encoders
        # Here we uses C module "_io" as actual type which pydantic will use is "_io._IOBase".
        import _io

        json_encoders = {_io._IOBase: lambda f: f.fileno()}

    # Provide default value as default_factory because stdout itself is not pickle-able.
    file: io.IOBase = Field(default_factory=lambda: sys.stdout)


# Format specific configs
configs = {RecordFormat.ChromeTrace: ChromeTraceConfig}


class profile:
    """Profiler context manager.

    Examples:
        >>> from furiosa.runtime.profiler import RecordFormat
        >>> with open("profile.json", "w") as f:
        >>>     with profile(format=RecordFormat.ChromeTrace, file=f) as profiler:
        >>>         # Profiler enabled from here
        >>>         with profiler.record("Inference"):
        >>>             ... # Profiler recorded with span named 'Inference'
    """

    def __init__(
        self,
        resource: Resource = Resource.ALL,
        format: RecordFormat = RecordFormat.ChromeTrace,
        **config: Any,
    ):
        """Create profiler context with specified arguments.

        Args:
            resource (Resource): Target resource to be profiled. e.g. CPU or NPU.
            format (RecordFormat): Profiler format. e.g. ChromeTrace.
            **config: Format specific config. You need to pass valid arguments for the format.

        Raises:
           pydantic.error_wrappers.ValidationError: Raise when config validation failed.
        """
        formats = ", ".join(map(str, RecordFormat))
        assert format in configs, f"Invalid format: {format}. Choose one of {formats}"

        self.resource = resource
        self.format = format
        self.config = configs[format](**config).json()

    def __enter__(
        self,
    ):
        """Enter profiler context.

        This enables profiler until this context exit.

        Returns:
           profile: Profiler context manager.

        Raises:
            errors.InvalidYamlException: Raise when config (YAML compatible expected) is invalid.
        """

        self.profiler = c_void_p(None)
        err = LIBNUX.profiler_enable(
            self.resource.value, self.format.value, self.config.encode(), byref(self.profiler)
        )

        if is_err(err):
            raise into_exception(err)

        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        """Exit profiler context.

        This disables the created profiler.
        """

        LIBNUX.profiler_disable(self.profiler)

    @contextmanager
    def record(
        self,
        name: str = "",
        warm_up=False,
    ):
        """Create profiler span with specified name.

        Args:
            name (str): Profiler record span name.
            warm_up (bool): If true, do not record profiler result, and just warm up.
        """

        span = LIBNUX.profiler_record_start(name.encode(), warm_up)
        yield
        LIBNUX.profiler_record_end(span)
