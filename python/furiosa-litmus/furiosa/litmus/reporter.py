from datetime import datetime
import hashlib
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from typing import Optional, Union
from zipfile import ZipFile

import distro
import onnx
import psutil
import yaml

from furiosa.device.sync import list_devices  # type: ignore
from furiosa.quantizer import version_dict as quantizer_version
from furiosa.runtime import version_dict as runtime_version
from furiosa.tools.compiler.api import version_dict as compiler_version


class Reporter:
    def __init__(self, dump_path: Optional[str] = None):
        if dump_path is None:
            self.dump_path: Union[None, str] = None
            self.compiler_log_path: Union[None, str] = None
            self.memory_analysis_path: Union[None, str] = None
            self.dot_graph_path: Union[None, str] = None
            self.compile_profiling_path: Union[None, str] = None
            self.trace_path: Union[None, str] = None
            self.runtime_profiling_path: Union[None, str] = None
        else:
            self.dump_path = f"{dump_path}-{int(time.time())}"
            self.tmp_dir = tempfile.TemporaryDirectory(prefix=self.dump_path, dir=".")
            self.meta_path = self.tmp_dir.name + "/meta.yaml"

            self.compiler_dir = self.tmp_dir.name + "/compiler"
            self.compiler_log_path = self.compiler_dir + "/compiler.log"
            self.memory_analysis_path = self.compiler_dir + "/memory-analysis.html"
            self.dot_graph_path = self.compiler_dir + "/model.dot"
            self.compile_profiling_path = self.compiler_dir + "/profiling.json"
            os.makedirs(self.compiler_dir)

            self.runtime_dir = self.tmp_dir.name + "/runtime"
            self.trace_path = self.runtime_dir + "/trace.json"
            self.runtime_profiling_path = self.runtime_dir + "/profiling.json"
            os.makedirs(self.runtime_dir)

    def __del__(self):
        if self.dump_path:
            self.tmp_dir.cleanup()

    @staticmethod
    def _get_md5_hash(filename):
        md5 = hashlib.md5()
        with open(filename, "rb") as f:
            md5.update(f.read())
        return md5.hexdigest()

    def create_meta_yaml(self, model_path):
        if self.dump_path is None:
            return

        sdk_meta = {
            "compiler": compiler_version(),
            "runtime": runtime_version(),
            "quantizer": quantizer_version(),
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
            "created_at": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
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

        with open(self.meta_path, "w") as f:
            yaml.dump(meta, f, sort_keys=False)

    def make_zip(self):
        if self.dump_path:
            with ZipFile(f"{self.dump_path}.zip", "w") as zip_object:
                for dir, _, filenames in os.walk(self.tmp_dir.name):
                    for filename in filenames:
                        zip_object.write(os.path.join(dir, filename))
            self.tmp_dir.cleanup()
