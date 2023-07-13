import os
import subprocess
import tempfile


def test_version():
    result = subprocess.run(["furiosa-compile", "--version"], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_compile(quantized_mnist_tflite):
    with tempfile.TemporaryDirectory() as tempdir:
        output = f"{tempdir}/output.enf"

        os.chdir(tempdir)
        result = subprocess.run(["furiosa-compile", quantized_mnist_tflite], capture_output=True)

        assert result.returncode == 0, result.stderr
        assert os.path.isfile(output)


def test_compile_with_output(quantized_mnist_tflite):
    with tempfile.TemporaryDirectory() as tempdir:
        output = f"{tempdir}/output.enf"
        result = subprocess.run(
            ["furiosa-compile", quantized_mnist_tflite, "-o", output],
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert os.path.isfile(output)


def test_compile_with_other_outputs(quantized_mnist_tflite):
    with tempfile.TemporaryDirectory() as tempdir:
        output = f"{tempdir}/output.enf"
        analyze_memory_output = f"{tempdir}/memory_analysis.html"
        dot_graph_output = f"{tempdir}/graph.dot"
        result = subprocess.run(
            [
                "furiosa-compile",
                quantized_mnist_tflite,
                "-o",
                output,
                "--dot-graph",
                dot_graph_output,
                "--analyze-memory",
                analyze_memory_output,
            ],
            capture_output=True,
        )

        assert result.returncode == 0, result.stderr
        assert os.path.isfile(output)
        assert os.path.isfile(analyze_memory_output)
        assert os.path.isfile(dot_graph_output)


def test_compile_with_target_npus(quantized_mnist_tflite):
    model = quantized_mnist_tflite

    with tempfile.TemporaryDirectory() as tempdir:
        output = f"{tempdir}/output.enf"

        result = subprocess.run(
            ["furiosa-compile", model, "-o", output, "--target-npu", "warboy"],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr

        result = subprocess.run(
            ["furiosa-compile", model, "-o", output, "--target-npu", "warboy-2pe"],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
