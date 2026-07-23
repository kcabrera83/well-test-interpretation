import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
