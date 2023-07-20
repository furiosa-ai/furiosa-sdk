import warnings

try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.profiler import *
else:
    from furiosa.runtime.legacy.profiler import *
