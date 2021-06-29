import importlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

consts = importlib.import_module('furiosa').consts

from . import argparser, commands, utils
from .exceptions import NoCommandException, CliError


class Session(object):
    def __init__(self):
        home = str(Path.home())
        load_dotenv('{}/.furiosa/config'.format(home), verbose=False, override=False)
        load_dotenv('{}/.furiosa/credential'.format(home), verbose=False, override=False)

        self.api_endpoint = os.environ.get(consts.FURIOSA_API_ENDPOINT_ENV)
        self.access_key_id = os.environ.get(consts.FURIOSA_ACCESS_KEY_ID_ENV)
        self.secret_key_access = os.environ.get(consts.SECRET_ACCESS_KEY_ENV)

        if self.api_endpoint is None:
            self.api_endpoint = consts.DEFAULT_API_ENDPOINT

    def ensure_apikey(self):
        if self.access_key_id is None or self.secret_key_access is None:
          raise CliError('FURIOSA_ACCESS_KEY_ID, FURIOSA_SECRET_ACCESS_KEY must be set', 1)


class CLIDriver(object):
    parser = None
    debug = False
    quiet = False

    commands = {'compile', 'perfeye', 'version'}

    def __init__(self, args, args_map):
        self.args = args
        self.args_map = args_map
        self.check_args()
        self.session = Session()

    def check_args(self):
        self.quiet = self.args.quiet
        self.debug = self.args.debug

        if self.args.command is None:
            raise NoCommandException()

    def run(self) -> int:
        if self.args.command == 'compile':
            cmd = commands.Compile(self.session, self.args, self.args_map)
        elif self.args.command == 'perfeye':
            cmd = commands.Perfeye(self.session, self.args, self.args_map)
        elif self.args.command == 'optimize':
            cmd = commands.Optimize(self.session, self.args, self.args_map)
        elif self.args.command == 'build_calibration_model':
            cmd = commands.BuildCalibrationModel(self.session, self.args, self.args_map)
        elif self.args.command == 'quantize':
            cmd = commands.Quantize(self.session, self.args, self.args_map)
        elif self.args.command == 'version':
            cmd = commands.Version(self.session, self.args, self.args_map)
        elif self.args.command == 'toolchain':
            if self.args.subcmd == 'list':
                cmd = commands.ToolchainList(self.session, self.args, self.args_map)
            else:
                raise CliError('toolchain requires one of following subcommands: list')
        elif self.args.command == 'validate':
            validate = utils.which("furiosa-validate")
            if validate is not None:
                cmd = commands.ValidateModel(None, self.args, self.args_map)
            else:
                raise CliError('Unknown command: {}'.format(self.args.command), 2)
        else:
            raise CliError('Unknown command: {}'.format(self.args.command), 2)

        return cmd.run()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    parser = argparser.create_argparser()
    args = parser.parse_args()
    args_map = vars(args)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        logging.debug("Arguments: {}".format(args))
        logging.debug("Arguments: {}".format(args_map))
        cli = CLIDriver(args, args_map)
        exit_code = cli.run()
    except NoCommandException as e:
        eprint('ERROR: {}\n'.format(e.message))
        parser.print_help()
        sys.exit(e.exit_code)
    except CliError as e:
        eprint('ERROR: {}\n'.format(e.message))
        sys.exit(e.exit_code)

    sys.exit(exit_code)
