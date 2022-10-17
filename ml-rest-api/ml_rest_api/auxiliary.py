import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from werkzeug.exceptions import BadRequest, NotFound


def parse_models(models_list):
    out = [{'id': m['id'], 'model_class': m['model'].__class__.__name__} for m in models_list]
    return out


def prepare_X(X):
    df = pd.DataFrame.from_records(X).values
    return df


def prepare_y(y):
    df = pd.Series(y).values
    return df


def train_model(model_class, X, y, **kwargs):
    if model_class == 'LinearRegression':
        mc = LinearRegression(**kwargs)
    elif model_class == 'RandomForestClassifier':
        mc = RandomForestClassifier(**kwargs)
    else:
        e = BadRequest('Unknown model class provided')
        raise e

    return mc.fit(X, y)


def predict_with_model(fitted_models, model_id, X):
    model = list(filter(lambda x: x['id'] == model_id, fitted_models))
    if len(model) == 0:
        e = NotFound('Model not found')
        raise e
    else:
        prediction = model[0]['model'].predict(X)
        return list(prediction)
