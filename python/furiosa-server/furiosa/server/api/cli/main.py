"""
Command line interface for FuriosaAI model server
"""

import enum
import logging

import typer

from ...server import ModelServer
from ...settings import ModelConfig, RESTServerConfig, ServerConfig
from ...utils.loader import load_model_config, load_server_config
from ...utils.thread import synchronous


class LogLevel(str, enum.Enum):
    """
    Log level enum for typer choice parameter
    """

    ERROR = "ERROR"
    INFO = "INFO"
    WARN = "WARN"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


@synchronous
async def start(
    log_level: LogLevel = LogLevel.INFO,
    model_path: str = typer.Option(None, help="Path to Model file (tflite, onnx are supported)"),
    model_name: str = typer.Option(None, help="Model name used in URL path"),
    model_version: str = typer.Option("default", help="Model version used in URL path"),
    host: str = typer.Option("0.0.0.0", help="IP address to bind"),
    http_port: int = typer.Option(8080, help="HTTP port to listen to requests"),
    model_config: typer.FileText = typer.Option(
        None, help="Path to a config file about models with specific configurations"
    ),
    server_config: typer.FileText = typer.Option(
        None, help="Path to Model file (tflite, onnx are supported)"
    ),
):
    """
    Start serving models from FuriosaAI model server
    """
    logging.basicConfig(level=log_level.value)

    if not model_config and not model_path:
        typer.echo("Model should be given either of model_path or model_config")
        raise typer.Exit(1)

    if server_config:
        config = load_server_config(server_config)
    else:
        rest_config = RESTServerConfig(port=http_port, debug=(log_level == LogLevel.DEBUG))
        config = ServerConfig(rest_server_config=rest_config)

    if model_config:
        model_configs = load_model_config(model_config)
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
