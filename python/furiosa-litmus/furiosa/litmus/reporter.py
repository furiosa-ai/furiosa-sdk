from datetime import datetime
import hashlib
import multiprocessing
import os
import platform
import subprocess as sp
import sys
import tempfile
import time
from typing import Optional
from zipfile import ZipFile

import distro
import onnx
import psutil
import yaml

from furiosa.device.sync import list_devices  # type: ignore
from furiosa.quantizer import __version__ as quantizer_version
from furiosa.runtime import __version__ as runtime_version


class DummyReporter:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, trackback):
        pass

    def get_compiler_log_path(self):
        return None

    def get_dot_graph_path(self):
        return None

    def get_memory_analysis_path(self):
        return None

    def get_compile_profiling_path(self):
        return None

    def get_trace_path(self):
        return None

    def get_runtime_profiling_path(self):
        return None

    def dump_meta_yaml(self, _):
        pass

    def make_zip(self):
        pass


class Reporter(DummyReporter):
    def __new__(cls, dump_path: Optional[str] = None):
        if dump_path:
            return super().__new__(cls)
        else:
            return DummyReporter()

    def __init__(self, dump_path: Optional[str]):
        self.zip_filename = f"{dump_path}-{int(time.time())}"
        self.tmp_dir = tempfile.TemporaryDirectory(prefix=self.zip_filename, dir=".")
        self.compiler_dir = self.tmp_dir.name + "/compiler"
        self.runtime_dir = self.tmp_dir.name + "/runtime"
        os.makedirs(self.compiler_dir)
        os.makedirs(self.runtime_dir)

    def __exit__(self, type, value, trackback):
        self.make_zip()
        self.tmp_dir.cleanup()

    def get_compiler_log_path(self):
        return self.compiler_dir + "/compiler.log"

    def get_dot_graph_path(self):
        return self.compiler_dir + "/model.dot"

    def get_memory_analysis_path(self):
        return self.compiler_dir + "/memory-analysis.html"

    def get_compile_profiling_path(self):
        return self.compiler_dir + "/profiling.json"

    def get_trace_path(self):
        return self.runtime_dir + "/trace.json"

    def get_runtime_profiling_path(self):
        return self.runtime_dir + "/profiling.json"

    @staticmethod
    def _get_md5_hash(filename):
        md5 = hashlib.md5()
        with open(filename, "rb") as f:
            md5.update(f.read())
        return md5.hexdigest()

    def dump_meta_yaml(self, model_path):
        sdk_meta = {
            "compiler": sp.run(
                ["furiosa-compiler", "--version"], capture_output=True, text=True
            ).stdout.strip(),
            "runtime": runtime_version.__dict__,
            "quantizer": quantizer_version.__dict__,
        }

        env_meta = {
            "python": sys.version.replace("\n", " "),
            "glibc": platform.libc_ver()[1],
            "libonnxruntime": onnx.version.version,
            "datetime": datetime.now(),
            "timezone": str(time.tzname),
        }

        os_meta = {
            "platform": platform.system(),
            "distrib": distro.name(),
            "distrib_release": distro.version(),
            "kernel": platform.release() + " / " + platform.version(),
        }

        cpu_meta = {
            "cores": multiprocessing.cpu_count(),
            "memory": psutil.virtual_memory().total,
        }

        model_meta = {
            "filename": os.path.basename(model_path),
            "size": os.path.getsize(model_path),
            "md5": Reporter._get_md5_hash(model_path),
            "created_at": {
                "datetime": datetime.fromtimestamp(os.path.getmtime(model_path)),
                "timezone": str(time.tzname),
            },
        }

        devices_meta = {}
        for npu in list_devices():
            firmware_version, firmware_rev = npu.firmware_version().split(", ")
            device_meta = {
                "serial_number": npu.device_sn(),
                "uuid": npu.device_uuid(),
                "pci_dev": npu.pci_dev(),
                "firmware": {"version": firmware_version, "rev": firmware_rev},
            }
            devices_meta[npu.name()] = device_meta

        meta = {
            "sdk": sdk_meta,
            "environment": env_meta,
            "os": os_meta,
            "cpu": cpu_meta,
            "model": model_meta,
            "devices": devices_meta,
            "compiler_config": {},
        }

        with open(self.tmp_dir.name + "/meta.yaml", "w") as f:
            yaml.dump(meta, f, sort_keys=False)

    def make_zip(self):
        with ZipFile(f"{self.zip_filename}.zip", "w") as zip_object:
            for dir, _, filenames in os.walk(self.tmp_dir.name):
                for filename in filenames:
                    zip_object.write(os.path.join(dir, filename))
