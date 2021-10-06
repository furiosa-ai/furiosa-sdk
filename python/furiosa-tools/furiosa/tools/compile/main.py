import argparse
import sys

from furiosa.tools import __version__
from furiosa.tools.compile._api import LIBCOMPILER, NuxLogLevel, compiler_version

DESCRIPTION: str = "FuriosaAI SDK Compiler for DNN models"

EXAMPLE: str = """example:
    # Compile foo.tflite into output.enf
    furiosa compile foo.tflite

    # Compile foo.onnx into foo.enf
    furiosa compile foo.onnx -o foo.enf

    # In addition to compilation, analyze the memory allocation and write the HTML report to bar.html
    furiosa compile foo.onnx -o foo.enf --analyze-memory bar.html

    # In addition to compilation, write the dot graph of the model graph into to bar.dot
    furiosa compile foo.onnx -o foo.enf --dot-graph bar.dot
"""

NPU_IDS = ["warboy", "warboy-2pe"]


class CommandArgError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__()


def set_ga_param(options, key: str, value: str):
    if key.lower() == "population_size":
        try:
            LIBCOMPILER.compiler_options_ga_population_size(options, int(value))
        except:
            raise CommandArgError("population_size must be a positive integer")

    elif key.lower() == "generation_limit":
        try:
            LIBCOMPILER.compiler_options_ga_generation_limit(options, int(value))
        except:
            CommandArgError("generation_limit must be a positive integer")

    elif key.lower() == "max_prefetch_size":
        try:
            LIBCOMPILER.compiler_options_ga_max_prefetch_size(options, int(value))
        except:
            raise CommandArgError("max_prefetch_size must be a positive integer")

    elif key.lower() == "nondeterministic":
        try:
            if not value.lower() in ['true', 'false']:
                raise RuntimeError()

            LIBCOMPILER.compiler_options_ga_nondeterministic(options, bool(value))
        except:
            raise CommandArgError("nondeterministic must be either 'true' or 'false'")

    elif key.lower() == "pin_tensors":
        try:
            if not value.lower() in ['true', 'false']:
                raise RuntimeError()

            LIBCOMPILER.compiler_options_ga_pin_tensors(options, bool(value))
        except:
            raise CommandArgError("pin_tensors must be either 'true' or 'false'")

    elif key.lower() == "init_tactic":
        try:
            if not value.lower() in ['random', 'heuristic']:
                raise RuntimeError()

            LIBCOMPILER.compiler_options_ga_pin_tensors(options, bool(value))
        except:
            raise CommandArgError("init_tactic must be either 'random' or 'heuristic'")

    else:
        raise CommandArgError(f"unknown genetic algorithm parameter: '{key}'")


def ga_options(options, list):
    for kv in list:
        parts = kv.split("=")
        if len(parts) != 2:
            raise CommandArgError("-ga parameters must be a form of KEY=VALUE; e.g., init_tactic=random")

        set_ga_param(options, parts[0], parts[1])


class CommandCompile():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EXAMPLE,
                                              formatter_class=argparse.RawDescriptionHelpFormatter)
        self.setup_arguments()
        self.parser.add_argument("--description", action="version", version=f"{DESCRIPTION}",
                                 help="show the command description")
        self.args = self.parser.parse_args()

    def setup_arguments(self):
        self.parser.add_argument('source', type=str,
                                 help='Path to source file (tflite, onnx, and other IR formats, such as dfg, cdfg, gir, lir)')
        self.parser.add_argument("--version", action="version",
                                 version=f"{compiler_version()} (wrapper: {__version__})")
        self.parser.add_argument('-o', dest='output', type=str,
                                 help='Writes output to <OUTPUT> (default: output.<TARGET_IR>)')
        self.parser.add_argument('--target-ir', type=str, default='enf',
                                 help='Target IR (dfg|cdfg|gir|lir|enf) (default: enf)')
        self.parser.add_argument('--target-npu', type=str, default='warboy',
                                 help='Target NPU: warboy (default), warboy-2pe')
        self.parser.add_argument('--analyze-memory', type=str,
                                 help='Analyzes the memory allocation and save the report to <ANALYZE_MEMORY>')
        self.parser.add_argument('--dot-graph', type=str,
                                 help='Filename to write DOT-formatted graph to')
        self.parser.add_argument('-v', '--verbose', action="store_true",
                                 help='Shows details about the compilation process')

        # Compile options
        self.parser.add_argument('--batch-size', type=int,
                                 help='Specifies the batch size which is effective when SOURCE is TFLite, '
                                      'ONNX, or DFG (default: 1)')
        self.parser.add_argument('--auto-batch-size', action="store_true",
                                 help='Find the optimal batch size automatically')
        self.parser.add_argument('--split-after-lower', action="store_true", help='Enable the split after lower')
        self.parser.add_argument('-ga', '--genetic-optimization',
                                 nargs='+', help='Use generic optimization with parameters')

    def run(self) -> int:
        envs = {}
        if self.args.verbose:
            LIBCOMPILER.enable_logging(NuxLogLevel.INFO)

        options = LIBCOMPILER.compiler_options_create()

        LIBCOMPILER.compiler_options_input(options, self.args.source.encode())

        if self.args.output is not None:
            output_file = self.args.output
        else:
            output_file = f"output.{self.args.target_ir}"

        LIBCOMPILER.compiler_options_output(options, output_file.encode())

        if self.args.target_ir:
            target_ir = self.args.target_ir
        else:
            target_ir = "enf"

        LIBCOMPILER.compiler_options_target_ir(options, target_ir.encode())

        if self.args.analyze_memory:
            LIBCOMPILER.compiler_options_memory_analysis(options, self.args.analyze_memory.encode())
        if self.args.dot_graph:
            LIBCOMPILER.compiler_options_dot_graph(options, self.args.dot_graph.encode())
        if self.args.batch_size:
            LIBCOMPILER.compiler_options_batch_size(options, self.args.batch_size)
        if self.args.auto_batch_size:
            LIBCOMPILER.compiler_options_auto_batch_size(options, True)
        if self.args.split_after_lower:
            LIBCOMPILER.compiler_options_split_after_lower(options, True)
        if self.args.target_npu:
            LIBCOMPILER.compiler_options_target_npu(options, self.args.target_npu.encode())

        if self.args.genetic_optimization:
            LIBCOMPILER.compiler_options_ga_optimization(options, True)
            ga_options(options, self.args.genetic_optimization)

        err = LIBCOMPILER.compiler_run(options)
        return err


def main():
    try:
        exit(CommandCompile().run())
    except CommandArgError as e:
        print(f"ERROR: {e.message}", file=sys.stderr)
        exit(255)


if __name__ == "__main__":
    main()
