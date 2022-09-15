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
    PandasDataFrame = 1


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


class PandasDataFrameConfig(BaseModel):
    """PandasDataFrame specific config.

    Attributes:
        file: file descriptor to write profile data. By default, memfd.
    """

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    # Provide default value as default_factory because stdout itself is not pickle-able.
    file: int


# Format specific configs
configs = {
    RecordFormat.ChromeTrace: ChromeTraceConfig,
    RecordFormat.PandasDataFrame: PandasDataFrameConfig,
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

        if format == RecordFormat.PandasDataFrame:
            if 'file' in config.keys():
                config['file'] = config['file'].fileno()
            else:
                config['file'] = os.memfd_create('profiling')

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
        self.df = pd.DataFrame()

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

        if self.format == RecordFormat.PandasDataFrame:
            config = json.loads(self.config)
            pos = os.lseek(config['file'], 0, os.SEEK_CUR)
            os.lseek(config['file'], 0, os.SEEK_SET)
            buf = os.read(config['file'], pos)
            self.df = pd.read_csv(
                io.StringIO(buf.decode()),
                dtype={
                    "cat": "string",
                    "dram_base": "Int64",
                    "pe_index": "Int64",
                    "execution_index": "Int64",
                    "instruction_index": "Int64",
                    "operator_index": "Int64",
                },
                keep_default_na=False,
            )
            self.df['dur'] = self.df['end'] - self.df['start']

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
        df = self.get_npu_pandas_dataframe()
        print(
            df[df["name"] != "Execution"]
            .groupby(["name"])
            .agg({"dur": "mean", "span_id": "count"})
            .sort_values(by=['dur'], ascending=False)
            .rename(columns={'dur': 'average elapsed(ns)', 'span_id': 'count'})
        )

    def print_npu_inference(self):
        df = self.get_npu_pandas_dataframe()
        execution = df[df["inst_idx"].isnull()]
        computation = (
            df[~df["name"].isin(["Execution", "Load", "Store"])][
                ["parent_span_id", "pe_idx", "exec_idx", "dur"]
            ]
            .groupby(["parent_span_id", "pe_idx", "exec_idx"], as_index=False)
            .sum()
        )
        merged = execution.merge(computation, left_on='span_id', right_on='parent_span_id')[
            ["trace_id", "span_id", "pe_idx_x", "exec_idx_x", "dur_x", "dur_y"]
        ]
        print(
            merged[["trace_id", "span_id", "pe_idx_x", "exec_idx_x", "dur_x", "dur_y"]].rename(
                columns={
                    'pe_idx_x': 'pe_idx',
                    'exec_idx_x': 'exec_idx',
                    'dur_x': 'NPU Total',
                    'dur_y': 'NPU Run',
                }
            )
        )

    def print_npu_inference_time_summary(self):
        df = self.get_npu_pandas_dataframe()
        execution = df[df["inst_idx"].isnull()]
        print("Total Elapsed: " + "{:,}".format(execution["dur"].sum()) + " ns")
        print("Average Elapsed: " + "{:,}".format(execution["dur"].mean()) + " ns")

    def print_external_operators(self):
        df = self.get_cpu_pandas_dataframe()
        print(df[~df["op_idx"].isnull()])

    def print_inferences(self):
        df = self.get_cpu_pandas_dataframe()
        print(df[df["name"] == "Inference"][["trace_id", "span_id", "tid", "dur"]])

    def print_inference_time_summary(self):
        df = self.get_cpu_pandas_dataframe()
        inferences = df[df["name"] == "Inference"]
        print("Total Elapsed: " + "{:,}".format(inferences["dur"].sum()) + " ns")
        print("Average Elapsed: " + "{:,}".format(inferences["dur"].mean()) + " ns")

    def get_pandas_dataframe(self):
        return self.df

    def get_pandas_dataframe_filter_with_cat(self, column, cat):
        return self.df[self.df[column] == cat]

    def get_cpu_pandas_dataframe(self):
        return self.get_pandas_dataframe_filter_with_cat("cat", "Nux")

    def get_npu_pandas_dataframe(self):
        return self.get_pandas_dataframe_filter_with_cat("cat", "NPU")

    def export_chrome_trace(self, filename):
        if self.df.empty:
            print("DataFrame is empty.")
            return

        with open(filename, 'w') as f:
            pid = os.getpid()
            items = []

            begin_time = self.df["start"].min()

            for idx, row in self.df.iterrows():
                obj = {}
                obj['name'] = row['name']
                obj['cat'] = row['cat']
                obj['ph'] = "X"
                obj['ts'] = (row['start'] - begin_time) / 1000.0
                obj['dur'] = row['dur'] / 1000.0
                obj['pid'] = pid
                obj['tid'] = row['thread.id']

                if row["cat"] == "Nux":
                    if not pd.isna(row['operator_index']):
                        obj['args'] = {'operator_index': str(row['operator_index'])}
                elif row["cat"] == "NPU":
                    obj['tid'] = row['pe_index'] + 10000000
                    obj['args'] = {
                        'pe_index': str(row['pe_index']),
                        'execution_index': str(row['execution_index']),
                    }
                    if not pd.isna(row['instruction_index']):
                        obj['args']['instruction_index'] = str(row['instruction_index'])

                items.append(obj)

            for row in self.get_npu_pandas_dataframe()['pe_index'].drop_duplicates():
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
