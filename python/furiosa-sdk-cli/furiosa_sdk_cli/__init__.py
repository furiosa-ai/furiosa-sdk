"""FuriosaSDK CLI"""
import importlib

__all__ = ["commands", "clidriver"]

from . import commands
from .clidriver import Session

utils = importlib.import_module("furiosa").utils
__version__ = utils.get_sdk_version(__name__)
