import pytest


@pytest.fixture(scope='package')
def data():
    """Returns the request body for the /predict endpoint excluding 'capital_gain'"""
    return {
        'age': 43,
        'workclass': 'Private',
        'fnlwgt': 12.2,
        'education': 'Doctorate',
        'education_num': 28,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        # 'capital_gain': 0, # NOTE: added/modified by the test
        'capital_loss': 0,
        'hours_per_week': 123,
        'native_country': 'United-States',
    }
