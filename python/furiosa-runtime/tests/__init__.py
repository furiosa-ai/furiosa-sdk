import os


def test_data(name: str) -> str:
    return os.path.dirname(__file__) + "/../../../tests/data/" + name
