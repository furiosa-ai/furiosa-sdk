"""
Tests to verify core REST APIs (Predict/Repository API)
"""

import asyncio
from asyncio.events import AbstractEventLoop
from typing import AsyncGenerator, Generator, List

from fastapi.testclient import TestClient
import pytest
import yaml

from furiosa.server.server import ModelServer
from furiosa.server.settings import ModelConfig, ServerConfig
from furiosa.server.utils.loader import load_model_config, load_server_config

MODEL_NAME = "mnist"
MODEL_VERSION = "default"
MODEL_CONFIG_EXAMPLE = {
    "model_config_list": [
        {
            "name": MODEL_NAME,
            "model": "samples/data/MNISTnet_uint8_quant.tflite",
            "version": MODEL_VERSION,
            "compiler_config": {"keep_unsignedness": True, "split_unit": 0},
            "platform": "nux",
        }
    ]
}

INVALID_MODEL_NAME = "NON_EXISTENT_MODEL"
INVALID_MODEL_VERSION = "NON_EXISTENT_VERSION"


@pytest.fixture(scope="module")
def event_loop() -> Generator[AbstractEventLoop, None, None]:
    """
    Create an instance of the default event loop for each test case

    Ref: https://github.com/pytest-dev/pytest-asyncio/issues/171
    """

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def server_config() -> ServerConfig:
    """
    Load server config from file
    """

    return load_server_config(open("samples/server_config_example.yaml"))


@pytest.fixture(scope="module")
def model_configs() -> List[ModelConfig]:
    """
    Load model configs from Python dict
    """

    return load_model_config(MODEL_CONFIG_EXAMPLE)


@pytest.fixture(scope="module")
async def client(
    server_config: ServerConfig, model_configs: List[ModelConfig]
) -> AsyncGenerator[TestClient, None]:
    """
    Create FastAPI TestClient from ModelServer created from server/model configs

    Note that this functions creates async fixture provided by pytest-asyncio

    Ref: https://fastapi.tiangolo.com/tutorial/testing/
    Ref: https://docs.pytest.org/en/6.2.x/fixture.html#fixture
    Ref: [asyncio](https://fastapi.tiangolo.com/advanced/async-tests/)
    """
    server = ModelServer(server_config, model_configs)
    await server.load()

    # Note that we are testing only REST server here
    yield TestClient(server._rest_server._app)


@pytest.fixture(scope="module")
def payload():
    return yaml.safe_load(open("samples/mnist_input_sample_01.json"))


def test_model_ready(client):
    response = client.get(f"/v2/models/{MODEL_NAME}/ready")
    assert response.status_code == 200

    response = client.get(f"/v2/models/{INVALID_MODEL_NAME}/ready")
    assert response.status_code == 400


def test_model_version_ready(client):
    response = client.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/ready")
    assert response.status_code == 200

    response = client.get(f"/v2/models/{MODEL_NAME}/versions/{INVALID_MODEL_VERSION}/ready")
    assert response.status_code == 400


def test_model_infer(client, payload):
    response = client.post(f"/v2/models/{MODEL_NAME}/infer", json=payload)
    assert response.status_code == 200

    response = client.post(f"/v2/models/{INVALID_MODEL_NAME}/infer", json=payload)
    assert response.status_code == 400


def test_model_version_infer(client, payload):
    response = client.post(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}/infer", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "id": None,
        "parameters": None,
        "outputs": [
            {
                "name": "0",
                "shape": [1, 10],
                "datatype": "UINT8",
                "parameters": None,
                "data": [0, 0, 0, 1, 0, 255, 0, 0, 0, 0],
            }
        ],
    }

    # Model version which was loaded first would be 'default' as well
    response = client.post(f"/v2/models/{MODEL_NAME}/versions/default/infer", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "id": None,
        "parameters": None,
        "outputs": [
            {
                "name": "0",
                "shape": [1, 10],
                "datatype": "UINT8",
                "parameters": None,
                "data": [0, 0, 0, 1, 0, 255, 0, 0, 0, 0],
            }
        ],
    }

    response = client.post(
        f"/v2/models/{INVALID_MODEL_NAME}/versions/{MODEL_VERSION}/infer", json=payload
    )
    assert response.status_code == 400

    response = client.post(
        f"/v2/models/{MODEL_NAME}/versions/{INVALID_MODEL_VERSION}/infer", json=payload
    )
    assert response.status_code == 400


def test_model_metadata(client):
    response = client.get(f"/v2/models/{MODEL_NAME}")
    assert response.status_code == 200
    assert response.json() == {
        "inputs": [],
        "name": MODEL_NAME,
        "outputs": [],
        "platform": "nux",
        "versions": [MODEL_VERSION],
    }

    response = client.get(f"/v2/models/{INVALID_MODEL_NAME}")
    assert response.status_code == 400


def test_model_version_metadata(client):
    response = client.get(f"/v2/models/{MODEL_NAME}/versions/{MODEL_VERSION}")
    assert response.status_code == 200
    assert response.json() == {
        "inputs": [],
        "name": MODEL_NAME,
        "outputs": [],
        "platform": "nux",
        "versions": [MODEL_VERSION],
    }

    response = client.get(f"/v2/models/{MODEL_NAME}/versions/{INVALID_MODEL_VERSION}")
    assert response.status_code == 400


def test_health_live(client):
    response = client.get("/v2/health/live")
    assert response.status_code == 200


def test_health_ready(client):
    response = client.get("/v2/health/ready")
    assert response.status_code == 200


def test_repository_index(client):
    response = client.post("/v2/repository/index", json={})
    assert response.status_code == 200
    assert response.json() == [
        {"name": "mnist", "version": MODEL_VERSION, "state": "READY", "reason": ""}
    ]

    response = client.post("/v2/repository/index", json={"ready": True})
    assert response.status_code == 200
    assert response.json() == [
        {"name": "mnist", "version": MODEL_VERSION, "state": "READY", "reason": ""}
    ]

    response = client.post("/v2/repository/index", json={"ready": False})
    assert response.status_code == 200
    assert response.json() == []


def test_repository_load(client, payload):
    # Unload model first
    response = client.post(f"/v2/repository/models/{MODEL_NAME}/unload")
    assert response.status_code == 200

    # Model state should be "UNAVAILABLE"
    response = client.post("/v2/repository/index", json={"ready": False})
    assert response.json() == [
        {"name": "mnist", "version": MODEL_VERSION, "state": "UNAVAILABLE", "reason": ""}
    ]

    # Infer should be failed
    response = client.post(f"/v2/models/{MODEL_NAME}/infer", json=payload)
    assert response.status_code == 400

    # Load unloaded model
    response = client.post(f"/v2/repository/models/{MODEL_NAME}/load")
    assert response.status_code == 200

    response = client.post("/v2/repository/index", json={"ready": True})
    # Model state should be "READY"
    assert response.json() == [
        {"name": "mnist", "version": MODEL_VERSION, "state": "READY", "reason": ""}
    ]

    # Infer should be succeed
    response = client.post(f"/v2/models/{MODEL_NAME}/infer", json=payload)
    assert response.status_code == 200
