try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.bench import *
else:
    # furiosa.runtime.bench is not available in legacy
    raise ModuleNotFoundError("No module named 'furiosa.runtime.bench")
