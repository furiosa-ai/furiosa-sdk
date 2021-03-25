"""FuriosaAI Python SDK"""
from . import utils

__version__ = utils.get_sdk_git_version()

# It's necessary to skip the documentation generation for generated codes
__pdoc__ = {
    "openapi": False
}
