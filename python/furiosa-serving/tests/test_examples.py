from pathlib import Path

from fastapi import status
from fastapi.testclient import TestClient

assets = Path(__file__).parent.parent / "examples/assets"


def test_examples():
    image_classify()
    number_classify()


def image_classify():
    from examples.image_classify import app

    with TestClient(app) as client, open(assets / "images/car.jpg", "rb") as img:
        response = client.post("/imagenet/infer", files={"image": img})

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result == {
        "car wheel": 148,
        "convertible": 148,
        "pickup": 152,
        "racer": 143,
        "sports car": 155,
    }


def number_classify():
    from examples.number_classify import app

    with TestClient(app) as client, open(assets / "images/1234567890.jpg", "rb") as img:
        response = client.post("/infer", files={"file": img})

    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert result == {
        "result": '1234567890',
    }
