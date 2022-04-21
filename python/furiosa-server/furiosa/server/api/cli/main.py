"""
Command line interface for FuriosaAI model server
"""

import enum
import logging
from typing import Any, Dict

from pydantic import ValidationError
import typer

from furiosa.common.thread import synchronous

from ...server import ModelServer
from ...settings import ModelConfig, RESTServerConfig, ServerConfig
from ...utils.loader import load_model_config, load_server_config


class LogLevel(str, enum.Enum):
    """
    Log level enum for typer choice parameter
    """

    ERROR = "ERROR"
    INFO = "INFO"
    WARN = "WARN"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


def _display_error_loc(error: Dict[str, Any]) -> str:
    return ' -> '.join(str(e) for e in error['loc'])


@synchronous
async def start(
    log_level: LogLevel = LogLevel.INFO,
    model_name: str = typer.Option(None, help="Model name"),
    model_path: str = typer.Option(None, help="Path to a model file (tflite, onnx are supported)"),
    model_version: str = typer.Option("default", help="Model version"),
    host: str = typer.Option("0.0.0.0", help="IPv4 address to bind"),
    http_port: int = typer.Option(8080, help="HTTP port to bind"),
    model_config: typer.FileText = typer.Option(None, help="Path to a model config file"),
    server_config: typer.FileText = typer.Option(None, help="Path to a server config file"),
):
    """
    Start serving models from FuriosaAI model server
    """
    logging.basicConfig(level=log_level.value)

    if not model_config:
        if not model_path:
            typer.echo("Missing option '--model-path'")
            raise typer.Exit(1)

        if not model_name:
            typer.echo("ERROR: missing option '--model-name'")
            raise typer.Exit(1)
    else:
        if model_path:
            typer.echo("ERROR: '--model-path' cannot be used with '--model-config'")
            raise typer.Exit(1)

        if model_name:
            typer.echo("ERROR: '--model-name' cannot be used with '--model-config'")
            raise typer.Exit(1)

    if server_config:
        config = load_server_config(server_config)
    else:
        rest_config = RESTServerConfig(
            host=host, port=http_port, debug=(log_level == LogLevel.DEBUG)
        )
        config = ServerConfig(rest_server_config=rest_config)

    if model_config:
        try:
            model_configs = load_model_config(model_config)
        except ValidationError as e:
            first_err = e.errors()[0]
            typer.echo(f"ERROR: {_display_error_loc(first_err)} {first_err['msg']}")
            raise typer.Exit(1)
    else:
        model_configs = [ModelConfig(model=model_path, name=model_name, version=model_version)]

    await ModelServer(config, model_configs).start()


def main():
    """
    Entry point for command line interface
    """
    typer.run(start)


if __name__ == "__main__":
    main()
