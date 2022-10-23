from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    """Tests root endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {
        "msg": "Called inference API root.",
        "success": True,
    }
