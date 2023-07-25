from __future__ import print_function

import logging as log
import pkgutil
import sys
from typing import Optional

from packaging.version import Version


class FuriosaVersionInfo:
    def __init__(self, version: Version):
        self.version = version.base_version
        self.stage = "dev" if version.is_devrelease else "release"
        assert version.local is not None
        self.hash = version.local.split(".")[0][1:]

    def __str__(self):
        return f"{self.version}-{self.stage} (rev: {self.hash[0:9]})"

    def __repr__(self):
        return f"FuriosaVersionInfo(({self.stage}, {self.version}, {self.hash}))"


def get_sdk_version(module) -> Optional[FuriosaVersionInfo]:
    """Returns the git commit hash representing the current version of the application."""
    sdk_version = None
    git_version = pkgutil.get_data(module, 'git_version.txt')

    assert git_version is not None

    try:
        version_string = str(git_version, encoding="UTF-8")
        sdk_version = FuriosaVersionInfo(Version(version_string))
    except Exception as e:  # pylint: disable=broad-except
        log.warn(e)

    return sdk_version


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def removeprefix(word: str, prefix: str) -> str:
    """Python 3.9 removeprefix().

    See https://docs.python.org/3/library/stdtypes.html#str.removeprefix
    """
    return word[len(prefix) :] if word.startswith(prefix) else word  # noqa: E203
