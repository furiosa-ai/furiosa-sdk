#!/usr/bin/env python
from distutils import log
import glob
import logging
import os
from os.path import dirname, relpath
import subprocess
from textwrap import wrap
from typing import Dict, List

from setuptools import Command, Distribution, Extension, find_namespace_packages, setup
from setuptools.command.develop import develop as develop_orig
from setuptools.command.install import install as install_orig

# Controls whether providers are installed from packages or directly from sources
# It is turned on by default in case of development environments such as Breeze
# And it is particularly useful when you add a new provider and there is no
# PyPI version to install the provider package from
INSTALL_EXTRAS_FROM_SOURCES = 'INSTALL_EXTRAS_FROM_SOURCES'

PREINSTALLED_PROVIDERS = []

logger = logging.getLogger(__name__)

version = '0.5.1'

my_dir = dirname(__file__)

here = os.path.abspath(os.path.dirname(__file__))

EXTRAS_REQUIREMENTS: Dict[str, List[str]] = {
    "server": ["furiosa-server~=" + version],
    "quantizer": ["furiosa-quantizer~=" + version],
    "validator": ["furiosa-model-validator~=" + version],
    "models": ["furiosa-models~=" + version],
}

# Requirements for all "user" extras (no devel). They are de-duplicated. Note that we do not need
# to separately add providers requirements - they have been already added as 'providers' extras above
_all_requirements = list(
    {req for extras_reqs in EXTRAS_REQUIREMENTS.values() for req in extras_reqs}
)

EXTRAS_REQUIREMENTS["full"] = _all_requirements


def git_version(version_: str) -> str:
    """
    Return a version to identify the state of the underlying git repo. The version will
    indicate whether the head of the current git-backed working directory is tied to a
    release tag or not : it will indicate the former with a 'release:{version}' prefix
    and the latter with a 'dev0' prefix. Following the prefix will be a sha of the current
    branch head. Finally, a "dirty" suffix is appended to indicate that uncommitted
    changes are present.
    :param str version_: Semver version
    :return: Found Furiosa SDK version in Git repo
    :rtype: str
    """
    try:
        import git

        try:
            repo = git.Repo(os.path.join(*[my_dir, '..', '..', '.git']))
        except git.NoSuchPathError:
            logger.warning('.git directory not found: Cannot compute the git version')
            return ''
        except git.InvalidGitRepositoryError:
            logger.warning('Invalid .git directory not found: Cannot compute the git version')
            return ''
    except ImportError:
        logger.warning('gitpython not found: Cannot compute the git version.')
        return ''
    if repo:
        sha = repo.head.commit.hexsha
        if repo.is_dirty():
            return f'.dev0+{sha}.dirty'
        # commit is clean
        return f'.release:{version_}+{sha}'
    return 'no_git_version'


def write_version(filename: str = os.path.join(*[my_dir, "furiosa/common", "git_version"])):
    """
    Write the Semver version + git hash to file, e.g. ".dev0+2f635dc265e78db6708f59f68e8009abb92c1e65".
    :param str filename: Destination file to write
    """
    text = f"{git_version(version)}"
    with open(filename, 'w') as file:
        file.write(text)


class Develop(develop_orig):
    """Forces removal of providers in editable mode."""

    def run(self):
        self.announce('Installing in editable mode. Uninstalling extra packages!', level=log.INFO)
        # We need to run "python3 -m pip" because it might be that older PIP binary is in the path
        # And it results with an error when running pip directly (cannot import pip module)
        # also PIP does not have a stable API so we have to run subprocesses ¯\_(ツ)_/¯
        try:
            installed_packages = (
                subprocess.check_output(["python3", "-m", "pip", "freeze"]).decode().splitlines()
            )
            furiosa_sdk_extras = [
                package_line.split("=")[0]
                for package_line in installed_packages
                if package_line.startswith("furiosa-sdk")
            ]
            self.announce(f'Uninstalling ${furiosa_sdk_extras}!', level=log.INFO)
            if furiosa_sdk_extras:
                subprocess.check_call(
                    ["python3", "-m", "pip", "uninstall", "--yes", *furiosa_sdk_extras]
                )
        except subprocess.CalledProcessError as e:
            self.announce(f'Error when uninstalling Furiosa SDK packages: {e}!', level=log.WARN)
        super().run()


def do_setup() -> None:
    write_version()
    setup_kwargs = {}
    setup(
        version=version,
        extras_require=EXTRAS_REQUIREMENTS,
        packages=find_namespace_packages(include=["furiosa.*"]),
        cmdclass={
            'develop': Develop,
        },
        entry_points={"console_scripts": ["furiosa=furiosa.cli:main"]},
        **setup_kwargs,
    )


if __name__ == "__main__":
    do_setup()
