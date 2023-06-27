from datetime import datetime
import hashlib
import multiprocessing
import os
from pathlib import Path
import platform
import sys
from zipfile import ZipFile

import distro
from furiosa_device.sync import list_devices
import onnx
import psutil
import yaml

from furiosa.quantizer import __dict_version__ as quantizer_version
from furiosa.runtime import __version__ as runtime_version
from furiosa.tools.compiler.api import version_dict as compiler_version


class Reporter:
    def __init__(self, dump_path: str = None):
        if dump_path:
            self.dump_path = Path(dump_path)
            self.meta_path = self.dump_path / 'meta.yaml'
            self.compiler_dir = self.dump_path / 'compiler'
            self.compiler_log_path = self.compiler_dir / 'compiler.log'
            self.memory_analysis_path = self.compiler_dir / 'memory-analysis.html'
            self.dot_graph_path = self.compiler_dir / 'model.dot'
            self.trace_path = self.compiler_dir / 'trace.json'
            self.profiling_path = self.compiler_dir / 'profiling.json'
            os.makedirs(self.compiler_dir, exist_ok=True)
        else:
            self.dump_path = None
            self.meta_path = None
            self.compiler_path = None
            self.compiler_log_path = None
            self.memory_analysis_path = None
            self.dot_graph_path = None
            self.trace_path = None
            self.profiling_path = None

    @staticmethod
    def _get_md5_hash(filename):
        md5 = hashlib.md5()
        with open(filename, 'rb') as f:
            md5.update(f.read())
        return md5.hexdigest()

    def create_meta_yaml(self, model_path):
        if self.dump_path is None:
            return

        sdk_meta = {
            'compiler': compiler_version(),
            # TODO
            # need git_short_hash, build_time
            'runtime': runtime_version,
            'quantizer': quantizer_version,
        }

        env_meta = {
            'python': sys.version.replace('\n', ' '),
            'glibc': platform.libc_ver()[1],
            'libonnxruntime': onnx.version.version,
        }

        os_meta = {
            'platform': platform.system(),
            'distrib': distro.name(),
            'distrib_release': distro.version(),
            'kernel': platform.release() + ' / ' + platform.version(),
        }

        cpu_meta = {
            'cores': multiprocessing.cpu_count(),
            'memory': psutil.virtual_memory().total,
        }

        model_meta = {
            'filename': os.path.basename(model_path),
            'size': os.path.getsize(model_path),
            'md5': Reporter._get_md5_hash(model_path),
            'created_at': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        }

        devices_meta = {}
        for npu in list_devices():
            firmware_version, firmware_rev = npu.firmware_version().split(', ')
            device_meta = {
                'serial_number': npu.device_sn(),
                'uuid': npu.device_uuid(),
                'pci_dev': npu.pci_dev(),
                'firmware': {'version': firmware_version, 'rev': firmware_rev},
            }
            devices_meta[npu.name()] = device_meta

        meta = {
            'sdk': sdk_meta,
            'environment': env_meta,
            'os': os_meta,
            'cpu': cpu_meta,
            'model': model_meta,
            'devices': devices_meta,
            'compiler_config': {},
        }

        with open(self.meta_path, 'w') as f:
            yaml.dump(meta, f, sort_keys=False)

    def make_zip(self):
        if self.dump_path:
            with ZipFile(f'{self.dump_path}.zip', 'w') as zip_object:
                for dir, _, filenames in os.walk(self.dump_path):
                    for filename in filenames:
                        zip_object.write(os.path.join(dir, filename))
