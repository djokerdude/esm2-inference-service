import pytest
from fastapi.testclient import TestClient
from src.server import app


@pytest.fixture(scope="session")
def client():
    """
    Start the full app once per test session — model load + FAISS index build
    happens here. All server tests share this client to avoid paying that
    startup cost multiple times.
    """
    with TestClient(app) as c:
        yield c
