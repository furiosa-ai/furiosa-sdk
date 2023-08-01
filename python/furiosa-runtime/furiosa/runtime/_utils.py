import os

from furiosa import runtime


def default_device() -> str:
    """Returns default device string for the current runtime.
    This function is for backward compatibility of the device argument for runtime."""
    if runtime.is_legacy:
        return os.environ.get("NPU_DEVNAME", "npu0pe0-1")
    else:
        return os.environ.get("FURIOSA_DEVICES", os.environ.get("NPU_DEVNAME", "warboy(2)*1"))
