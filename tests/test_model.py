from numpy import ndarray
from model.ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)
from sklearn.ensemble import RandomForestClassifier


def test_compute_model_metrics():
    """Tests compute_model_metrics() output types."""
    x, y, z = compute_model_metrics(
        [1, 0, 1, 0],
        [1, 0, 1, 0],
    )
    assert isinstance(x, (float, int))
    assert isinstance(y, (float, int))
    assert isinstance(z, (float, int))


def test_inference(model_RF, data):
    """Tests compute_model_metrics() output types."""
    preds = inference(model_RF, data.X)
    assert isinstance(preds, ndarray)


def test_train_model(data):
    """Tests compute_model_metrics() output types."""
    model = train_model(data.X, data.y)
    assert isinstance(model, RandomForestClassifier)
