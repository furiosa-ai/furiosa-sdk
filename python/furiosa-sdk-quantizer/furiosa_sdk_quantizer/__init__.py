__all__ = ["frontend", "interfaces", "ir", "scripts"]

import logging
import pkgutil
from . import frontend, interfaces, ir, scripts

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def get_sdk_git_version():
    """Returns the git commit hash representing the current version of the application."""
    git_version = None
    try:
        git_version = str(pkgutil.get_data('furiosa_sdk_quantizer', 'git_version'), encoding="UTF-8")
    except Exception as err:  # pylint: disable=broad-except
        LOG.debug(err)

    return git_version


__version__ = get_sdk_git_version()