"""FuriosaAI Python SDK"""
import importlib
from . import utils

_furiosa_sdk_runtime = importlib.util.find_spec("furiosa_sdk_runtime")
if _furiosa_sdk_runtime is not None:
    import furiosa_sdk_runtime as runtime

_furiosa_sdk_quantizer = importlib.util.find_spec("furiosa_sdk_quantizer")
if _furiosa_sdk_quantizer is not None:
    import furiosa_sdk_quantizer as quantizer

__version__ = utils.get_sdk_git_version()

# It's necessary to skip the documentation generation for generated codes
__pdoc__ = {
    "openapi": False
}
