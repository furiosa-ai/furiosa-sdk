__all__ = ["frontend", "interfaces", "ir"]

from . import frontend, interfaces, ir
import importlib

utils = importlib.import_module('furiosa').utils
__version__ = utils.get_sdk_version(__name__)
