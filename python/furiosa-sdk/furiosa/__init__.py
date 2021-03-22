"""FuriosaAI Python SDK"""
from . import utils

import importlib
furiosa_runtime = importlib.util.find_spec("furiosa.runtime")
if furiosa_runtime is not None:
    import furiosa.runtime

__version__ = utils.get_sdk_git_version()
