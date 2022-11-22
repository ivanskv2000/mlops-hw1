from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from werkzeug.exceptions import BadRequest, NotFound, UnprocessableEntity
import os
import json
import joblib

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


def get_model_metadata(id, m):
    """
    Return the saved models list in an appropriate format
    """

    def filter_hp(hp: dict):
        allowed_hp = [i["hyperparameters"] for i in model_classes["classes"]]
        allowed_hp = sum(allowed_hp, [])
        return {k: v for k, v in hp.items() if k in allowed_hp}

    out = {
        "id": id,
        "model_class": m.__class__.__name__,
        "hyperparameters": filter_hp(m.get_params()),
    }

    return out


def parse_models(path_to_models):
    metadata_files = [
        os.path.join(path_to_models, file)
        for file in os.listdir(path_to_models)
        if file.endswith(".json")
    ]
    metadata_out = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as cf:
            md = json.load(cf)
            metadata_out.append(md)

    return metadata_out


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
def predict_with_model(model_id, path_to_models, X):
    """
    Return predictions
    """
    model = [
        os.path.join(path_to_models, file)
        for file in os.listdir(path_to_models)
        if file == f"{model_id}.joblib"
    ]
    if len(model) == 0:
        e = NotFound("Model not found.")
        raise e
    else:
        model = joblib.load(model[0])
        prediction = model.predict(X)

        if model.__class__.__name__ == "RandomForestClassifier":
            prediction = prediction.astype(str)

        return list(prediction)


@model_errors_handler
def re_train(model_id, path_to_models, X, y):
    """
    Re-train an existing model
    """
    model = [
        os.path.join(path_to_models, file)
        for file in os.listdir(path_to_models)
        if file == f"{model_id}.joblib"
    ]

    if len(model) == 0:
        e = NotFound("Model not found.")
        raise e
    else:
        model = joblib.load(model[0])
        return model.fit(X, y)


def save_model(model_id, fitted_model, path_to_models):
    model_metadata = get_model_metadata(model_id, fitted_model)

    model_pickle_path = os.path.join(path_to_models, f"{model_id}.joblib")
    model_metadata_path = os.path.join(path_to_models, f"{model_id}.json")

    joblib.dump(fitted_model, model_pickle_path)

    with open(model_metadata_path, "w") as outfile:
        json.dump(model_metadata, outfile, indent=4)


def delete_model(model_id, path_to_models):
    model = [
        os.path.join(path_to_models, file)
        for file in os.listdir(path_to_models)
        if file == f"{model_id}.joblib"
    ]

    if len(model) == 0:
        e = NotFound("Model not found.")
        raise e
    else:
        os.remove(model[0])
        os.remove(os.path.join(path_to_models, f"{model_id}.json"))
