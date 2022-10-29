# Script to train machine learning model.

from io import StringIO

import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics, 
    inference, 
    train_model, 
    save_model,
    evaluate_slices,
)

# load in the data from remote storage
df = pd.read_csv(StringIO(dvc.api.read('../data/census_clean.csv')))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save the model.
model = train_model(X_train, y_train)
save_model(model, model.__class__.__name__)

# save transformers
save_model(encoder, encoder.__class__.__name__)
save_model(lb, lb.__class__.__name__)

# evalute and print results
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"The precision score for {model.__class__.__name__} is: {precision}")
print(f"The recall score for {model.__class__.__name__} is: {recall}")
print(f"The fbeta score for {model.__class__.__name__} is: {fbeta}")

# evaluate performance on 'workclass' slice
evaluate_slices(model, test, X_test, y_test, cat_features[0])
