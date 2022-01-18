import argparse
import sys
from typing import Dict

from furiosa.tools import __version__
from furiosa.tools.compiler.api import LIBCOMPILER, CompilerApiError, compile, version_string

DESCRIPTION: str = "FuriosaAI SDK Compiler for DNN models"

EXAMPLE: str = """example:
    # Compile foo.tflite into output.enf
    furiosa compile foo.tflite

    # Compile foo.onnx into foo.enf
    furiosa compile foo.onnx -o foo.enf
"""

NPU_IDS = ["warboy", "warboy-2pe"]


class CommandArgError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__()


def convert_ga_param(key: str, value: str) -> object:
    if key.lower() == "population_size":
        try:
            return int(value)
        except:
            raise CommandArgError("population_size must be a positive integer")

    elif key.lower() == "generation_limit":
        try:
            return int(value)
        except:
            CommandArgError("generation_limit must be a positive integer")

    elif key.lower() == "max_prefetch_size":
        try:
            return int(value)
        except:
            raise CommandArgError("max_prefetch_size must be a positive integer")

    elif key.lower() == "nondeterministic":
        if value.lower() in ['true', 'false']:
            return bool(value)
        else:
            raise CommandArgError("nondeterministic must be either 'true' or 'false'")

    elif key.lower() == "pin_tensors":
        if value.lower() in ['true', 'false']:
            return bool(value)
        else:
            raise CommandArgError("pin_tensors must be either 'true' or 'false'")

    elif key.lower() == "init_tactic":
        if value.lower() in ['random', 'heuristic']:
            return str(value)
        else:
            raise CommandArgError("init_tactic must be either 'random' or 'heuristic'")

    else:
        raise CommandArgError(f"unknown genetic algorithm parameter: '{key}'")


def ga_options(ga_params_str: str) -> Dict[str, object]:

    ga_params: Dict[str, object] = {}

    for kv in [part.strip() for part in ga_params_str.split(",")]:
        parts = kv.split("=")
        if len(parts) == 2:
            ga_params[parts[0]] = convert_ga_param(parts[0], parts[1])
        else:
            raise CommandArgError(
                "-ga parameters must be a form of KEY1=VALUE1,KEY2=VALUE2; e.g., init_tactic=random"
            )

    return ga_params


class CommandCompile:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=DESCRIPTION,
            epilog=EXAMPLE,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.setup_arguments()
        self.parser.add_argument(
            "--description",
            action="version",
            version=f"{DESCRIPTION}",
            help="show the command description",
        )
        self.args = self.parser.parse_args()

    def setup_arguments(self):
        self.parser.add_argument(
            'source',
            type=str,
            help='Path to source file (tflite, onnx, and other IR formats, such as dfg, cdfg, gir, lir)',
        )
        self.parser.add_argument(
            "--version", action="version", version=f"{version_string()} (wrapper: {__version__})"
        )
        self.parser.add_argument(
            '-o',
            dest='output',
            type=str,
            help='Writes output to <OUTPUT> (default: output.<TARGET_IR>)',
        )
        self.parser.add_argument(
            '--target-npu',
            type=str,
            default='warboy',
            help='Target NPU: warboy (default), warboy-2pe',
        )
        self.parser.add_argument(
            '-v',
            '--verbose',
            action="store_true",
            help='Shows details about the compilation process',
        )

        # Compile options
        self.parser.add_argument(
            '--batch-size',
            type=int,
            help='Specifies the batch size which is effective when SOURCE is TFLite, '
            'ONNX, or DFG (default: 1)',
        )
        self.parser.add_argument(
            '--auto-batch-size',
            action="store_true",
            help='Find the optimal batch size automatically',
        )

    def run(self) -> int:
        return compile(
            self.args.source,
            output=self.args.output,
            batch_size=self.args.batch_size,
            auto_batch_size=self.args.auto_batch_size,
            target_npu=self.args.target_npu,
            verbose=self.args.verbose,
        )


def main():
    try:
        exit(CommandCompile().run())
    except (CommandArgError, CompilerApiError) as e:
        print(f"ERROR: {e.message}", file=sys.stderr)
        exit(255)


if __name__ == "__main__":
    main()
