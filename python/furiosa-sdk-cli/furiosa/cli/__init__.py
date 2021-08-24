"""FuriosaSDK CLI"""
__all__ = ['commands', 'clidriver']

from furiosa.common.utils import get_sdk_version

from . import commands

__version__ = get_sdk_version(__name__)

