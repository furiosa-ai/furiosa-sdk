try:
    import package_extras
except ModuleNotFoundError:
    from furiosa.native_runtime.sync import *
else:
    # furiosa.runtime.sync is not available in legacy
    raise ModuleNotFoundError("No module named 'furiosa.runtime.sync")
