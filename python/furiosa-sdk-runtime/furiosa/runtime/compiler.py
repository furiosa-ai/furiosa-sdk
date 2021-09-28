"""Compiler module which compiles model images or intermediate IRs into executable NPU programs"""
import ctypes
import logging
import os
import random
import string
from ctypes import byref, c_uint64
from datetime import datetime
from pathlib import Path
from typing import Union, Dict

import yaml

from . import envs
from ._api import LIBNUX
from .errors import is_ok, into_exception

LOG = logging.getLogger(__name__)


def _read_file(path: Union[str, Path]):
    with open(path, 'rb') as file:
        contents = file.read()
        return contents


def _model_image(model: Union[bytes, str, Path]) -> bytes:
    if isinstance(model, bytes):
        model_image = model
    elif isinstance(model, (str, Path)):
        model_image = _read_file(model)
    else:
        raise TypeError("'model' must be str or bytes, but it was " + repr(type(model)))

    return model_image


def _generate_suffix(length: int) -> str:
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def _generate_unique_log_filename(log_dir: Path) -> Path:
    log_time = datetime.now().strftime(f"%Y%m%d%H%M%S")
    while True:
        log_filename = f"compile-{log_time}-{_generate_suffix(6)}.log"
        path = Path(f"{log_dir}/{log_filename}")
        try:
            path.touch(mode=0o644, exist_ok=False)
        except FileExistsError:
            continue

        return path


def compile_model(model: Union[bytes, str, Path],
                  device: str,
                  compile_config: Dict[str, object] = None,
                  target_ir: str = 'ENF') -> bytes:
    """Compile a model image into an executable NPU program

    Args:
        model (bytes, str, or Path): a bytearray of a model or a path of model image
        device (str): NPU device name (e.g., npu0pe0, npu0pe0-1)
        compile_config (dict): compiler configuration
        target_ir (str): IR type to be emitted (e.g., DFG, GIR, LIR, ENF)

    Returns:
        A byte array of an Executable NPU Program (ENF)
    """
    log_path = None
    if envs.is_compile_log_enabled():
        log_dir = Path(envs.log_dir())
        log_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        log_path = _generate_unique_log_filename(log_dir)
        LOG.info(f"Saving the compiler log into {log_path}")

    if target_ir is not None:
        target_ir = target_ir.encode()
    if compile_config is not None:
        compile_config = yaml.dump(compile_config).encode()
    if log_path is not None:
        log_path_encoded = str(log_path).encode()
    else:
        log_path_encoded = None

    model_image = _model_image(model)
    buf_ptr = ctypes.POINTER(ctypes.c_uint8)()
    buf_len = c_uint64(0)
    err = LIBNUX.nux_model_compile(device.encode(), model_image, len(model_image),
                                   compile_config,
                                   target_ir,
                                   log_path_encoded,
                                   byref(buf_ptr), byref(buf_len))
    if is_ok(err):
        try:
            array_type = ctypes.c_uint8 * buf_len.value
            buffer = bytearray(array_type.from_address(ctypes.addressof(buf_ptr.contents)))
        except Exception:
            raise RuntimeError('fail to read compiled program from buffer')
        finally:
            LIBNUX.nux_buffer_destroy(buf_ptr, buf_len)

        return bytes(buffer)
    else:
        if log_path:
            width, _ = os.get_terminal_size()
            LOG.error("=" * width)
            LOG.error(f"Please check the compile log file at {log_path}\n"
                      f"If it is a bug, please report the log file to https://github.com/furiosa-ai/furiosa-sdk/issues")
            LOG.error("=" * width)
        raise into_exception(err)
