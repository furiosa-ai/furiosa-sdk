"""FuriosaAI Python SDK"""
from . import utils

__version__ = utils.get_sdk_version(__name__)

# It's necessary to skip the documentation generation for generated codes
__pdoc__ = {
    "openapi": False
}
