import importlib.util

_furiosa_sdk_runtime = importlib.util.find_spec("furiosa_sdk_runtime")

if _furiosa_sdk_runtime is not None:
    from furiosa_sdk_runtime.model import *

del importlib