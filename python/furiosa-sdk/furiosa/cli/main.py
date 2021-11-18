import argparse
import os
import subprocess
import sys
from typing import Dict

from furiosa.cli.utils import which
from furiosa.common import __version__

COMMAND_LIST = {
    "furiosa-compile": "FuriosaAI SDK Compiler for DNN models",
    "furiosa-server": "RESTful/GRPC Serving framework for DNN models",
    "furiosa-validate": "Validator to check if DNN models work in FuriosaAI SDK",
}


class Command(object):
    def __init__(self, command: str, path: str, descriptor: str):
        self.command = command
        self.subcommands = [x.replace("_", "-") for x in command.split("-")][1:]
        self.path = path
        self.description = descriptor


class CommandRegistry(object):
    def __init__(self, check_exitence=True):
        found_commands = _register_commands(COMMAND_LIST, check_exitence)

        self.commands = {}
        for cmd in found_commands:
            self.commands[cmd.subcommands[0]] = cmd


def _register_command(cmd: str):
    cmd = Command(cmd)
    return cmd


def _register_commands(allow_commands: Dict[str, str], check_existence=True):
    found_commands = []
    for cmd in allow_commands.keys():
        path = which(cmd)
        if path:
            found_commands.append(Command(cmd, which(cmd), allow_commands[cmd]))

    return found_commands


def _register_subcommands(parser: argparse.ArgumentParser, registry: CommandRegistry):
    subparsers = parser.add_subparsers(dest='subcommand', help=f'sub-command help')

    for subcmd in registry.commands.keys():
        cmd = registry.commands[subcmd]
        parser = subparsers.add_parser(subcmd, help=cmd.description, add_help=False)


def execute(registry: CommandRegistry, args, remainings) -> int:
    envs = os.environ
    subcmd = args.subcommand
    command = [registry.commands[subcmd].path]
    command.extend(remainings)
    process = subprocess.Popen(
        command, env=envs, stdout=sys.stdout, stderr=sys.stderr, close_fds=True
    )
    exitcode = process.wait()
    return exitcode


def _parse_arguments(registry: CommandRegistry):
    parser = argparse.ArgumentParser(
        description="FuriosaAI SDK CLI",
        epilog="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"FuriosaAI SDK CLI v{__version__}")
    _register_subcommands(parser, registry)

    return parser


def _run(registry, parser):
    args, remainings = parser.parse_known_args()
    if args.subcommand:
        exit(execute(registry, args, remainings))
    else:
        parser.print_help()


def main():
    registry = CommandRegistry()
    parser = _parse_arguments(registry)
    _run(registry, parser)


if __name__ == "__main__":
    main()
