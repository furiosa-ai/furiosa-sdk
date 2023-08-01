import os

from furiosa import runtime


def test_default_device():
    if runtime.is_legacy:
        # Remove NPU_DEVNAME
        os.environ.pop('NPU_DEVNAME', None)
        assert runtime._utils.default_device() == 'npu0pe0-1'
    else:
        # Remove FURIOSA_DEVICES & NPU_DEVNAME
        os.environ.pop('NPU_DEVNAME', None)
        os.environ.pop('FURIOSA_DEVICES', None)
        assert runtime._utils.default_device() == 'warboy(2)*1'
