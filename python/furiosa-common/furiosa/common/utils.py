from __future__ import print_function

import logging as log
import pkgutil
import sys
import typing


class FuriosaVersionInfo:
    def __init__(self, parts: typing.Iterable[str]):
        self.stage, self.version, self.hash = parts

    def __str__(self):
        return f"{self.version} (rev: {self.hash[0:9]})"

    def __repr__(self):
        return f"FuriosaVersionInfo(({self.stage}, {self.version}, {self.hash}))"


def get_sdk_version(module) -> FuriosaVersionInfo:
    """Returns the git commit hash representing the current version of the application."""
    git_version = None
    try:
        verion_string = str(pkgutil.get_data(module, 'git_version'), encoding="UTF-8")
        git_version = FuriosaVersionInfo(verion_string.split(":"))
    except Exception as e:  # pylint: disable=broad-except
        log.debug(e)

    return git_version


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
