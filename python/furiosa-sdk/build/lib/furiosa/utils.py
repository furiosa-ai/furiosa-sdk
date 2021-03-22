import pkgutil
import logging as log


def get_sdk_git_version():
    """Returns the git commit hash representing the current version of the application."""
    git_version = None
    try:
        git_version = str(pkgutil.get_data('furiosa-sdk', 'git_version'), encoding="UTF-8")
    except Exception as e:  # pylint: disable=broad-except
        log.debug(e)

    return git_version
