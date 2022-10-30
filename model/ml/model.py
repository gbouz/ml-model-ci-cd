from joblib import dump, load
from numpy import concatenate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def save_model(model, name):
    """
    Saves a machine learning model localy.
    Inputs
    ------
    model
        Trained machine learning model.
    name : str
        File name.
    Returns
    -------
    None
    """
    dump(model, f'{name}.joblib')
    return


def load_model(name):
    """
    Loads a machine learning model and returns it.
    Inputs
    ------
    name : str
        File name.
    Returns
    -------
    model
        Trained machine learning model.
    """
    return load(f'{name}.joblib')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=3, 
        min_samples_leaf=4, 
        random_state=0,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X, categorical_features=[], encoder=None):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    encoder : OneHotEncoder
        OneHotEncoder used in production to transformation 
        the data to the right format.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if encoder:
        X_categorical = X[categorical_features].values
        X_continuous = X.drop(*[categorical_features], axis=1)
        X_categorical = encoder.transform(X_categorical)
        X = concatenate([X_continuous, X_categorical], axis=1)
    return model.predict(X)


def evaluate_slices(model, df, X, y, selected_feature):
    """ Evaluate model in slices of a given feature and save results.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    df : pd.DataFrame
        Dataframe used to slice the data in X and y.
    X : np.array
        Data used for prediction.
    y : np.array
        Data used for evaluating the predictions.
    selected_feature : str
        Column name of the selected feature to slice upon.
    Returns
    -------
    None
    """
    slice_output = []
    slice_tags = df[selected_feature].unique()
    slice_output.append(
        f"Evaluation of {model.__class__.__name__} on the following "
        f"'{selected_feature}' slices: "
    )
    slice_output.append(slice_tags)
    slice_output.append('-----------------------------------------------')
    for tag in slice_tags:
        slice_output.append(f" - Slice '{tag}' - ")
        mapper = df[selected_feature]==tag
        if mapper.sum() < 1:
            slice_output[-1] += '[NO OBSERVATIONS - skipping evaluation]'
            slice_output.append('-----------------------------------------------')
            continue
        preds = inference(model, X[mapper])
        precision, recall, fbeta = compute_model_metrics(y[mapper], preds)
        slice_output.append(f"Precision: {precision}")
        slice_output.append(f"Recall: {recall}")
        slice_output.append(f"Fbeta: {fbeta}")
        n_true = y[mapper].sum()
        slice_output.append(
            f"(n total obs: {y[mapper].shape[0]}, n true obs {n_true} ,"
            f" n false obs {y[mapper].shape[0] - n_true})"
        )
        slice_output.append('-----------------------------------------------')
    print('===============================================')
    with open('slice_output.txt', 'w') as filehandle:
        for output in slice_output:
            print(output)
            filehandle.write(f'{output}\n')
