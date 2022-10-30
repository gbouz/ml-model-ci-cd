from typing import Literal

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from model.ml.model import inference, load_model

app = FastAPI()


cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


class CensusData(BaseModel):
    age: float
    workclass: Literal[
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 
        'State-gov', 'Without-pay', 'Never-worked']
    fnlwgt: float
    education: Literal[
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 
        'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 
        'Doctorate', '5th-6th', 'Preschool']
    education_num: float
    marital_status: Literal[
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 
        'Married-spouse-absent', 'Married-AF-spouse']
    occupation: Literal[
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
        'Armed-Forces']
    relationship: Literal[
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    race: Literal['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    sex: Literal['Male', 'Female']
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: Literal[
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
        'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
        'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 
        'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 
        'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
        'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                'age': 43,
                'workclass': 'Private',
                'fnlwgt': 123075,
                'education': 'Doctorate',
                'education_num': 16,
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': 'United-States',
            }
        }



@app.get("/")
async def get_items():
    return {
        "msg": "Called inference API root.",
        "success": True,
    }


@app.post("/predict/")
async def predict(data: CensusData):
    encoder = load_model('model/OneHotEncoder')
    lb = load_model('model/LabelBinarizer')
    model = load_model('model/RandomForestClassifier')
    preds = inference(
        model, 
        pd.DataFrame([data.dict()]), 
        categorical_features=cat_features, 
        encoder=encoder,
    )
    return {
        "predictions": {
            "salary": {
                "bracket": lb.inverse_transform(preds).tolist(),
            },
        },
        "success": True,
    }