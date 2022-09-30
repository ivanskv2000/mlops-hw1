import numpy as np
import pandas as pd


def parse_models(models_list):
    out = [{'id': m['id'], 'model_class': m['model'].__class__.__name__} for m in models_list]
    return out


def prepare_X(X):
    df = pd.DataFrame.from_records(X).values
    return df


def prepare_y(y):
    df = pd.Series(y).values
    return df