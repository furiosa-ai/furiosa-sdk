#!/usr/bin/env python
import glob
import logging
import os
import subprocess
from distutils import log
from os.path import dirname, relpath
from textwrap import wrap
from typing import Dict, List

from setuptools import setup, Command, Distribution, Extension, find_namespace_packages
from setuptools.command.develop import develop as develop_orig
from setuptools.command.install import install as install_orig
from setuptools_rust import Binding, RustExtension

# Controls whether providers are installed from packages or directly from sources
# It is turned on by default in case of development environments such as Breeze
# And it is particularly useful when you add a new provider and there is no
# PyPI version to install the provider package from
INSTALL_EXTRAS_FROM_SOURCES = 'INSTALL_EXTRAS_FROM_SOURCES'

PREINSTALLED_PROVIDERS = []

logger = logging.getLogger(__name__)

version = '0.1.0.dev1'

my_dir = dirname(__file__)

here = os.path.abspath(os.path.dirname(__file__))

rust_extensions = None
if os.getenv('BUILD_NPU_TOOLS', 'False') == '1':

    if os.getenv('NPU_TOOLS_PATH') is None:
        print('NPU_TOOLS_PATH is not set')
        exit(1)

    nux_cargo_path = "{}/crates/nux/Cargo.toml".format(os.getenv('NPU_TOOLS_PATH'))
    rust_extensions = [RustExtension(
        "nux/_api/nux",
        path=nux_cargo_path,
        binding=Binding.NoBinding,
        debug=False,
        features=["use_web_api"],
    )]


class FuriosaSdkDistribution(Distribution):
    """
    The setuptools.Distribution subclass with FuriosaAI Sdk specific behaviour
    """

    def parse_config_files(self, *args, **kwargs):  # pylint: disable=signature-differs
        """
        Ensure that when we have been asked to install providers from sources
        that we don't *also* try to install those providers from PyPI.
        Also we should make sure that in this case we copy provider.yaml files so that
        Providers manager can find package information.
        """
        super().parse_config_files(*args, **kwargs)
        if os.getenv(INSTALL_EXTRAS_FROM_SOURCES) == 'true':
            self.install_requires = [  # noqa  pylint: disable=attribute-defined-outside-init
                req for req in self.install_requires if not req.startswith('furiosa-sdk-')
            ]
            provider_yaml_files = glob.glob("furiosa/extras/**/extra.yaml", recursive=True)
            for provider_yaml_file in provider_yaml_files:
                provider_relative_path = relpath(provider_yaml_file, os.path.join(my_dir, "furiosa"))
                self.package_data['furiosa'] = [provider_relative_path]
        else:
            self.install_requires.extend(
                [get_provider_package_from_package_id(package_id) for package_id in PREINSTALLED_PROVIDERS]
            )

dss_dependencies = [
]

runtime_dependencies = [
    'cffi',
    'numpy'
]

EXTRAS_REQUIREMENTS: Dict[str, List[str]] = {
    "runtime": runtime_dependencies,
    #"dss": dss_dependencies,
}

# Requirements for all "user" extras (no devel). They are de-duplicated. Note that we do not need
# to separately add providers requirements - they have been already added as 'providers' extras above
_all_requirements = list({req for extras_reqs in EXTRAS_REQUIREMENTS.values() for req in extras_reqs})

EXTRAS_REQUIREMENTS["all"] = _all_requirements

class ListExtras(Command):
    """
    List all available extras
    Registered as cmdclass in setup() so it can be called with ``python setup.py list_extras``.
    """

    description = "List available extras"
    user_options: List[str] = []

    def initialize_options(self):
        """Set default values for options."""

    def finalize_options(self):
        """Set final values for options."""

    def run(self):  # noqa
        """List extras."""
        print("\n".join(wrap(", ".join(EXTRAS_REQUIREMENTS.keys()), 100)))


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


def write_version(filename: str = os.path.join(*[my_dir, "furiosa", "git_version"])):
    """
    Write the Semver version + git hash to file, e.g. ".dev0+2f635dc265e78db6708f59f68e8009abb92c1e65".
    :param str filename: Destination file to write
    """
    text = f"{git_version(version)}"
    with open(filename, 'w') as file:
        file.write(text)


class CleanCommand(Command):
    """
    Command to tidy up the project root.
    Registered as cmdclass in setup() so it can be called with ``python setup.py extra_clean``.
    """

    description = "Tidy up the project root"
    user_options: List[str] = []

    def initialize_options(self):
        """Set default values for options."""

    def finalize_options(self):
        """Set final values for options."""

    @staticmethod
    def rm_all_files(files: List[str]):
        """Remove all files from the list"""
        for file in files:
            try:
                os.remove(file)
            except Exception as e:  # noqa pylint: disable=broad-except
                logger.warning("Error when removing %s: %s", file, e)

    def run(self):
        """Remove temporary files and directories."""
        os.chdir(my_dir)
        self.rm_all_files(glob.glob('./build/*'))
        self.rm_all_files(glob.glob('./**/__pycache__/*', recursive=True))
        self.rm_all_files(glob.glob('./**/*.pyc', recursive=True))
        self.rm_all_files(glob.glob('./dist/*'))
        self.rm_all_files(glob.glob('./*.egg-info/*'))


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
                subprocess.check_call(["python3", "-m", "pip", "uninstall", "--yes", *furiosa_sdk_extras])
        except subprocess.CalledProcessError as e:
            self.announce(f'Error when uninstalling Furiosa SDK packages: {e}!', level=log.WARN)
        super().run()


class Install(install_orig):
    """Forces installation of providers from sources in editable mode."""

    def run(self):
        self.announce('Standard installation. Providers are installed from packages', level=log.INFO)
        super().run()


PROVIDERS_REQUIREMENTS: Dict[str, List[str]] = {
    'runtime': runtime_dependencies,
}

ALL_PROVIDERS = list(PROVIDERS_REQUIREMENTS.keys())

def get_provider_package_from_package_id(package_id: str):
    """
    Builds the name of provider package out of the package id provided/

    :param package_id: id of the package (like amazon or microsoft.azure)
    :return: full name of package in PyPI
    """
    package_suffix = package_id.replace(".", "-")
    return f"furiosa-sdk-{package_suffix}"


def add_provider_packages_to_extra_requirements(extra: str, providers: List[str]) -> None:
    """
    Adds provider packages as requirements to extra. This is used to add provider packages as requirements
    to the "bulk" kind of extras. Those bulk extras do not have the detailed 'extra' requirements as
    initial values, so instead of replacing them (see previous function) we can extend them.

    :param extra: Name of the extra to add providers to
    :param providers: list of provider ids
    """
    EXTRAS_REQUIREMENTS[extra].extend(
        [get_provider_package_from_package_id(package_name) for package_name in providers]
    )

def add_all_provider_packages() -> None:
    for provider in ALL_PROVIDERS:
        replace_extra_requirement_with_provider_packages(provider, [provider])

    add_provider_packages_to_extra_requirements("all", ALL_PROVIDERS)


def replace_extra_requirement_with_provider_packages(extra: str, providers: List[str]) -> None:
    """
    Replaces extra requirement with provider package. The intention here is that when
    the provider is added as dependency of extra, there is no need to add the dependencies
    separately. This is not needed and even harmful, because in case of future versions of
    the provider, the requirements might change, so hard-coding requirements from the version
    that was available at the release time might cause dependency conflicts in the future.

    Say for example that you have salesforce provider with those deps:

    { 'salesforce': ['simple-salesforce>=1.0.0', 'tableauserverclient'] }

    Initially ['salesforce'] extra has those requirements and it works like that when you install
    it when INSTALL_PROVIDERS_FROM_SOURCES is set to `true` (during the development). However, when
    the production installation is used, The dependencies are changed:

    { 'salesforce': ['apache-airflow-providers-salesforce'] }

    And then, 'apache-airflow-providers-salesforce' package has those 'install_requires' dependencies:
            ['simple-salesforce>=1.0.0', 'tableauserverclient']

    So transitively 'salesforce' extra has all the requirements it needs and in case the provider
    changes it's dependencies, they will transitively change as well.

    In the constraint mechanism we save both - provider versions and it's dependencies
    version, which means that installation using constraints is repeatable.

    :param extra: Name of the extra to add providers to
    :param providers: list of provider ids
    """
    EXTRAS_REQUIREMENTS[extra] = [
        get_provider_package_from_package_id(package_name) for package_name in providers
    ]

def do_setup() -> None:
    """
    Perform the FuriosaAI SDK package setup.
    Most values come from setup.cfg, only the dynamically calculated ones are passed to setup
    function call. See https://setuptools.readthedocs.io/en/latest/userguide/declarative_config.html
    """
    setup_kwargs = {}

    def include_extra_namespace_packages_when_installing_from_sources() -> None:
        """
        When installing providers from sources we install all namespace packages found below airflow,
        including airflow and provider packages, otherwise defaults from setup.cfg control this.
        The kwargs in setup() call override those that are specified in setup.cfg.
        """
        if os.getenv(INSTALL_EXTRAS_FROM_SOURCES) == 'true':
            setup_kwargs['packages'] = find_namespace_packages(include=['furiosa*', '../furiosa*'])

    include_extra_namespace_packages_when_installing_from_sources()
    if os.getenv(INSTALL_EXTRAS_FROM_SOURCES) == 'true':
        print("Installing providers from sources. Skip adding providers as dependencies")
    else:
        add_all_provider_packages()

    write_version()

    setup(
        distclass=FuriosaSdkDistribution,
        version=version,
        extras_require=EXTRAS_REQUIREMENTS,
        cmdclass={
            'extra_clean': CleanCommand,
            'list_extras': ListExtras,
            'install': Install,
            'develop': Develop,
        },
        scripts=['bin/furiosa'],
        **setup_kwargs,
    )


if __name__ == "__main__":
    do_setup()
