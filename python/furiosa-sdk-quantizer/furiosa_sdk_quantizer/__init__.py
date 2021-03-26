__all__ = ["frontend", "interfaces", "ir", "scripts"]

from . import frontend, interfaces, ir, scripts
import importlib

utils = importlib.import_module('furiosa').utils
__version__ = utils.get_sdk_version(__name__)