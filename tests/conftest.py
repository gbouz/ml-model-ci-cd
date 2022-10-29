from dataclasses import dataclass

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainData:
    """Class that defines train data and labels."""
    X: np.ndarray
    y: np.ndarray


@pytest.fixture
def data():
    """Returns the train data"""
    return TrainData(
        *make_classification(
            n_samples=1000, n_features=4, n_informative=2, 
            n_redundant=0, random_state=0, shuffle=False,
        )
    )


@pytest.fixture
def model_RF(data):
    """Returns a trained RandomForestClassifier."""
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(data.X, data.y)
    return clf
