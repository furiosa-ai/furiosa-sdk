from __future__ import annotations

from contextlib import contextmanager
from ctypes import byref, c_void_p
from enum import Flag, IntEnum, auto
import io
import json
import os
import sys
from types import TracebackType
from typing import Any, Type

import pandas as pd
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
    CsvTrace = 1


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


class CsvTraceConfig(BaseModel):
    """CsvTrace specific config.

    Attributes:
        cpu_file: file descriptor to write CPU profile data. By default, memfd.
        npu_file: file descriptor to write NPU profile data. By default, memfd.
    """

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    # Provide default value as default_factory because stdout itself is not pickle-able.
    cpu_file: int
    npu_file: int


# Format specific configs
configs = {
    RecordFormat.ChromeTrace: ChromeTraceConfig,
    RecordFormat.CsvTrace: CsvTraceConfig,
}


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

        if format == RecordFormat.CsvTrace:
            if Resource.CPU in resource:
                if 'cpu_file' in config.keys():
                    config['cpu_file'] = config['cpu_file'].fileno()
                else:
                    config['cpu_file'] = os.memfd_create('cpu_profiling')
            else:
                config['cpu_file'] = -1

            if Resource.NPU in resource:
                if 'npu_file' in config.keys():
                    config['npu_file'] = config['npu_file'].fileno()
                else:
                    config['npu_file'] = os.memfd_create('npu_profiling')
            else:
                config['npu_file'] = -1

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
        self.df = {
            "cpu": pd.DataFrame(),
            "npu": pd.DataFrame(),
        }

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

        if self.format == RecordFormat.CsvTrace:
            config = json.loads(self.config)
            if 'cpu_file' in config.keys() and config['cpu_file'] >= 0:
                pos = os.lseek(config['cpu_file'], 0, os.SEEK_CUR)
                os.lseek(config['cpu_file'], 0, os.SEEK_SET)
                buf = os.read(config['cpu_file'], pos)
                self.df['cpu'] = pd.read_csv(io.StringIO(buf.decode()), dtype={"op_idx": "Int64"})
                self.df['cpu']['dur'] = self.df['cpu']["end"] - self.df['cpu']["start"]

            if 'npu_file' in config.keys() and config['npu_file'] >= 0:
                pos = os.lseek(config['npu_file'], 0, os.SEEK_CUR)
                os.lseek(config['npu_file'], 0, os.SEEK_SET)
                buf = os.read(config['npu_file'], pos)
                self.df['npu'] = pd.read_csv(
                    io.StringIO(buf.decode()),
                    dtype={"pe_idx": "Int64", "exec_idx": "Int64", "inst_idx": "Int64"},
                )
                self.df['npu']['dur'] = self.df['npu']["end"] - self.df['npu']["start"]

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

    def print_npu_operators(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.NPU in self.resource:
                print(
                    self.df['npu'][self.df['npu']["name"] != "Execution"]
                    .groupby(["name"])
                    .agg({"dur": "mean", "span_id": "count"})
                    .sort_values(by=['dur'], ascending=False)
                    .rename(columns={'dur': 'average elapsed(ns)', 'span_id': 'count'})
                )

    def print_npu_inference(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.NPU in self.resource:
                execution = self.df['npu'][self.df['npu']["inst_idx"].isnull()]
                computation = (
                    self.df['npu'][~self.df['npu']["name"].isin(["Execution", "Load", "Store"])][
                        ["parent_span_id", "pe_idx", "exec_idx", "dur"]
                    ]
                    .groupby(["parent_span_id", "pe_idx", "exec_idx"], as_index=False)
                    .sum()
                )
                merged = execution.merge(computation, left_on='span_id', right_on='parent_span_id')[
                    ["trace_id", "span_id", "pe_idx_x", "exec_idx_x", "dur_x", "dur_y"]
                ]
                print(
                    merged[
                        ["trace_id", "span_id", "pe_idx_x", "exec_idx_x", "dur_x", "dur_y"]
                    ].rename(
                        columns={
                            'pe_idx_x': 'pe_idx',
                            'exec_idx_x': 'exec_idx',
                            'dur_x': 'NPU Total',
                            'dur_y': 'NPU Run',
                        }
                    )
                )

    def print_npu_inference_time_summary(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.NPU in self.resource:
                execution = self.df['npu'][self.df['npu']["inst_idx"].isnull()]
                print("Total Elapsed: " + "{:,}".format(execution["dur"].sum()) + " ns")
                print("Average Elapsed: " + "{:,}".format(execution["dur"].mean()) + " ns")

    def print_external_operators(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.CPU in self.resource:
                print(self.df['cpu'][~self.df['cpu']["op_idx"].isnull()])

    def print_inferences(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.CPU in self.resource:
                print(
                    self.df['cpu'][self.df['cpu']["name"] == "Inference"][
                        ["trace_id", "span_id", "tid", "dur"]
                    ]
                )

    def print_inference_time_summary(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.CPU in self.resource:
                inferences = self.df['cpu'][self.df['cpu']["name"] == "Inference"]
                print("Total Elapsed: " + "{:,}".format(inferences["dur"].sum()) + " ns")
                print("Average Elapsed: " + "{:,}".format(inferences["dur"].mean()) + " ns")

    def get_cpu_pandas_dataframe(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.CPU in self.resource:
                return self.df['cpu']

        raise Exception("No DataFrame acquired.")

    def get_npu_pandas_dataframe(self):
        if self.format == RecordFormat.CsvTrace:
            if Resource.NPU in self.resource:
                return self.df['npu']

        raise Exception("No DataFrame acquired.")

    def export_chrome_trace(self, filename):
        if self.format == RecordFormat.CsvTrace:
            with open(filename, 'w') as f:
                pid = os.getpid()
                items = []

                cpu_begin_time = (
                    self.df['cpu']["start"].min() if not self.df['cpu'].empty else sys.maxsize
                )
                npu_begin_time = (
                    self.df['npu']["start"].min() if not self.df['npu'].empty else sys.maxsize
                )
                begin_time = min(cpu_begin_time, npu_begin_time)

                for idx, row in self.df['cpu'].iterrows():
                    obj = {}
                    obj['name'] = row['name']
                    obj['cat'] = "Nux"
                    obj['ph'] = "X"
                    obj['ts'] = (row['start'] - begin_time) / 1000.0
                    obj['dur'] = row['dur'] / 1000.0
                    obj['pid'] = pid
                    obj['tid'] = row['tid']
                    if not pd.isna(row['op_idx']):
                        obj['args'] = {'op_idx': row['op_idx']}
                    items.append(obj)

                for idx, row in self.df['npu'].iterrows():
                    obj = {}
                    obj['name'] = row['name']
                    obj['cat'] = "NPU"
                    obj['ph'] = "X"
                    obj['ts'] = (row['start'] - begin_time) / 1000.0
                    obj['dur'] = row['dur'] / 1000.0
                    obj['pid'] = pid
                    obj['tid'] = row['pe_idx'] + 10000000
                    obj['args'] = {'pe_idx': row['pe_idx'], 'exec_idx': row['exec_idx']}
                    if not pd.isna(row['inst_idx']):
                        obj['args']['inst_idx'] = row['inst_idx']
                    items.append(obj)

                for row in self.df['npu']['pe_idx'].drop_duplicates():
                    value = row.item()
                    obj = {}
                    obj['ph'] = "M"
                    obj['name'] = "thread_name"
                    obj['pid'] = pid
                    obj['tid'] = value + 10000000
                    obj['args'] = {'name': "NPU " + str(value)}
                    items.append(obj)

                json.dump(items, f)
                print("Export trace event format completed.")
