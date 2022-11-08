import argparse
import logging
import os
import sys
from typing import Dict

from furiosa.tools import __version__
from furiosa.tools.compiler.api import CompilerApiError, VersionInfo, compile, version_string

logging.basicConfig(level=os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper())

DESCRIPTION: str = f"FuriosaAI SDK Compiler (ver. {__version__.version})"

EXAMPLE: str = """example:
    # Compile foo.tflite into output.enf
    furiosa compile foo.tflite

    # Compile foo.onnx into foo.enf
    furiosa compile foo.onnx -o foo.enf

    # In addition to compilation, analyze the memory allocation and write the HTML report to bar.html
    furiosa compile foo.onnx -o foo.enf --analyze-memory bar.html

    # In addition to compilation, write the dot graph of the model graph into to bar.dot
    furiosa compile foo.onnx -o foo.enf --dot-graph bar.dot
    
    # Set the genetic algorithm parameters for optimization
    furiosa compile foo.onnx -o foo.enf -ga population_size=100,generation_limit=500
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
        full_version = f"""furiosa-compiler: {version_string()}\nfuriosa-tools: {__version__}\n"""
        self.parser.add_argument(
            'source',
            type=str,
            help='Path to source file (tflite, onnx, and other IR formats, such as dfg, cdfg, gir, lir)',
        )
        self.parser.add_argument("--version", action="version", version=full_version)
        self.parser.add_argument(
            '-o',
            dest='output',
            type=str,
            help='Writes output to <OUTPUT> (default: output.<TARGET_IR>)',
        )
        self.parser.add_argument(
            '--target-ir',
            type=str,
            default='enf',
            help='Target IR (dfg|cdfg|gir|lir|enf) (default: enf)',
        )
        self.parser.add_argument(
            '--target-npu',
            type=str,
            default='warboy-2pe',
            help='Target NPU: warboy, warboy-2pe (default)',
        )
        self.parser.add_argument(
            '--analyze-memory',
            type=str,
            help='Analyzes the memory allocation and save the report to <ANALYZE_MEMORY>',
        )
        self.parser.add_argument(
            '--dot-graph', type=str, help='Filename to write DOT-formatted graph to'
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
        self.parser.add_argument(
            '--split-after-lower', action="store_true", help='Enable the split after lower'
        )
        self.parser.add_argument(
            '-ga',
            '--genetic-optimization',
            type=str,
            help='Use generic optimization with parameters',
        )

    def run(self) -> int:
        compiler_version = VersionInfo()
        print(
            f"furiosa-compiler {compiler_version.version} (rev. {compiler_version.git_hash}), "
            f"furiosa-tools {__version__.version} (rev. {__version__.hash[0:9]})",
            file=sys.stderr,
        )

        ga_params = None
        if self.args.genetic_optimization:
            ga_params = ga_options(self.args.genetic_optimization)

        if self.args.output is None:
            output = f"output.{self.args.target_ir}"
        else:
            output = self.args.output

        return compile(
            self.args.source,
            output_path=output,
            target_ir=self.args.target_ir,
            dot_graph=self.args.dot_graph,
            analyze_memory=self.args.analyze_memory,
            batch_size=self.args.batch_size,
            auto_batch_size=self.args.auto_batch_size,
            ga_params=ga_params,
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
