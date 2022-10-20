import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from werkzeug.exceptions import BadRequest, NotFound, UnprocessableEntity


model_classes = {
    'classes': [
        {
            'name': 'LinearRegression',
            'predictors': ['numeric'],
            'target': ['numeric'],
            'hyperparameters': ['fit_intercept']
        },
        {
            'name': 'RandomForestClassifier',
            'predictors': ['numeric', 'categorical'],
            'target': ['categorical'],
            'hyperparameters': ['n_estimators', 'criterion', 'max_depth']
        }
    ]
}


def parse_models(models_list):
    def filter_hp(hp: dict):
        allowed_hp = [i['hyperparameters'] for i in model_classes['classes']]
        allowed_hp = sum(allowed_hp, [])
        return {k: v for k, v in hp.items() if k in allowed_hp}

    out = [
        {
            'id': m['id'],
            'model_class': m['model'].__class__.__name__,
            'hyperparameters': filter_hp(m['model'].get_params())
            } for m in models_list
        ]
    return out


def prepare_X(X):
    try:
        df = pd.DataFrame.from_records(X).values
    except TypeError:
        raise UnprocessableEntity('Wrong data format provided.')

    return df


def prepare_y(y):
    try:
        df = pd.Series(y).values
    except TypeError:
        raise UnprocessableEntity('Wrong data format provided.')

    return df


def model_errors_handler(func):
    def wrapper(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
            return out
        except ValueError:
            raise UnprocessableEntity('Wrong data type provided.')
        except TypeError:
            raise BadRequest('Unknown hyperparameter.')

    return wrapper


@model_errors_handler
def train_model(model_class, X, y, **kwargs):
    if model_class == 'LinearRegression':
        mc = LinearRegression(**kwargs)
    elif model_class == 'RandomForestClassifier':
        mc = RandomForestClassifier(**kwargs)
    else:
        e = BadRequest('Unknown model class provided.')
        raise e

    return mc.fit(X, y)


@model_errors_handler
def predict_with_model(fitted_models, model_id, X):
    model = list(filter(lambda x: x['id'] == model_id, fitted_models))
    if len(model) == 0:
        e = NotFound('Model not found.')
        raise e
    else:
        prediction = model[0]['model'].predict(X)
        return list(prediction)


@model_errors_handler
def re_train(fitted_models, model_id, X, y):
    model = list(filter(lambda x: x['id'] == model_id, fitted_models))
    if len(model) == 0:
        e = NotFound('Model not found.')
        raise e
    else:
        return model[0]['model'].fit(X, y)
