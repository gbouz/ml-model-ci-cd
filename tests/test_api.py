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


def test_predict_high_income(data):
    """Tests prediction for an expected high income data point."""
    data['capital_gain'] = 123123
    response = client.post('/predict/', json=data)
    assert response.status_code == 200
    assert response.json() == {
        'predictions': {'salary': {'bracket': ['>50K']}}, 'success': True}


def test_predict_low_income(data):
    """Tests prediction for an expected low income data point."""
    data['capital_gain'] = 0
    response = client.post('/predict/', json=data)
    assert response.status_code == 200
    assert response.json() == {
        'predictions': {'salary': {'bracket': ['<=50K']}}, 'success': True}
