import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from werkzeug.exceptions import BadRequest, NotFound, UnprocessableEntity

model_classes = {
    "classes": [
        {
            "name": "LinearRegression",
            "predictors": ["numeric"],
            "target": ["numeric"],
            "hyperparameters": ["fit_intercept"],
        },
        {
            "name": "RandomForestClassifier",
            "predictors": ["numeric", "categorical"],
            "target": ["categorical"],
            "hyperparameters": ["n_estimators", "criterion", "max_depth"],
        },
    ]
}


def parse_models(models_list):
    """
    Return the saved models list in an appropriate format
    """

    def filter_hp(hp: dict):
        allowed_hp = [i["hyperparameters"] for i in model_classes["classes"]]
        allowed_hp = sum(allowed_hp, [])
        return {k: v for k, v in hp.items() if k in allowed_hp}

    out = [
        {
            "id": m["id"],
            "model_class": m["model"].__class__.__name__,
            "hyperparameters": filter_hp(m["model"].get_params()),
        }
        for m in models_list
    ]
    return out


def prepare_X(X):
    """
    Convert json data to a Pandas dataframe
    """
    try:
        df = pd.DataFrame.from_records(X).values
    except TypeError:
        raise UnprocessableEntity("Wrong data format provided.")

    return df


def prepare_y(y):
    """
    Convert json array to a Pandas dataframe
    """
    try:
        df = pd.Series(y).values
    except TypeError:
        raise UnprocessableEntity("Wrong data format provided.")

    return df


def model_errors_handler(func):
    """
    A decorator used for handling exceptions
    which may occur during model training or prediction
    """

    def wrapper(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
            return out
        except ValueError:
            raise UnprocessableEntity("Wrong data type provided.")
        except TypeError:
            raise BadRequest("Unknown hyperparameter.")

    return wrapper


@model_errors_handler
def train_model(model_class, X, y, **kwargs):
    """
    Train a model
    """
    if model_class == "LinearRegression":
        mc = LinearRegression(**kwargs)
    elif model_class == "RandomForestClassifier":
        mc = RandomForestClassifier(**kwargs)
    else:
        e = BadRequest("Unknown model class provided.")
        raise e

    return mc.fit(X, y)


@model_errors_handler
def predict_with_model(fitted_models, model_id, X):
    """
    Return predictions
    """
    model = list(filter(lambda x: x["id"] == model_id, fitted_models))
    if len(model) == 0:
        e = NotFound("Model not found.")
        raise e
    else:
        prediction = model[0]["model"].predict(X)

        if model[0]["model"].__class__.__name__ == "RandomForestClassifier":
            prediction = prediction.astype(str)

        return list(prediction)


@model_errors_handler
def re_train(fitted_models, model_id, X, y):
    """
    Re-train an existing model
    """
    model = list(filter(lambda x: x["id"] == model_id, fitted_models))
    if len(model) == 0:
        e = NotFound("Model not found.")
        raise e
    else:
        return model[0]["model"].fit(X, y)
