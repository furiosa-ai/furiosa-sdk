from datetime import datetime
import logging
from pathlib import Path
import random
import string
from typing import Union

from . import envs
from ._util import eprint

LOG = logging.getLogger(__name__)


def _read_file(path: Union[str, Path]):
    with open(path, 'rb') as file:
        contents = file.read()
        return contents


def _model_image(model: Union[bytes, str, Path]) -> bytes:
    if isinstance(model, bytes):
        model_image = model
    elif isinstance(model, (str, Path)):
        model_image = Path(model).read_bytes()
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


def generate_compiler_log_path() -> Path:
    """Generate a log path for compilation log"""
    if envs.is_compile_log_enabled():
        compiler_log_path = Path(envs.log_dir())
        compiler_log_path.mkdir(mode=0o755, parents=True, exist_ok=True)
        compiler_log_path = _generate_unique_log_filename(compiler_log_path)
        eprint(f"Saving the compilation log into {compiler_log_path}")
        return compiler_log_path
    else:
        return None
