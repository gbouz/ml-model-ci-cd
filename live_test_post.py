import json
import requests

base_url = 'https://ml-model-ci-cd.herokuapp.com/'

data = {
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
    'capital_gain': 123123,
    'capital_loss': 0,
    'hours_per_week': 123,
    'native_country': 'United-States',
}

response = requests.post(
    base_url + 'predict/',
    json=data,
)

print('Response status code:', response.status_code)
print('Response payload:')
print(json.dumps(response.json(), indent=4))