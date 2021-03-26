import importlib.util

_furiosa_sdk_quantizer = importlib.util.find_spec("furiosa_sdk_quantizer")

if _furiosa_sdk_quantizer is not None:
    from furiosa_sdk_quantizer import *
    from furiosa_sdk_quantizer import __version__
