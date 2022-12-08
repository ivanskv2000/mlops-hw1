from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from werkzeug.exceptions import BadRequest, NotFound, UnprocessableEntity
import os
import json
import pickle
from .data_helpers import session, MlModel
from sqlalchemy.exc import NoResultFound

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


def get_model_metadata(m):
    """
    Return the saved models list in an appropriate format
    """

    def filter_hp(hp: dict):
        allowed_hp = [i["hyperparameters"] for i in model_classes["classes"]]
        allowed_hp = sum(allowed_hp, [])
        return {k: v for k, v in hp.items() if k in allowed_hp}

    out = {
        "model": pickle.dumps(m),
        "model_class": m.__class__.__name__,
        "hyperparameters": filter_hp(m.get_params()),
    }

    return out


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


def database_errors_handler(func):
    """
    A decorator used for handling exceptions
    which may occur during model training or prediction
    """

    def wrapper(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
            return out
        except NoResultFound:
            raise NotFound("Model not found.")
        #except TypeError:
        #    raise BadRequest("Unknown hyperparameter.")

    return wrapper


@database_errors_handler
def parse_models():
    db_entries = session.query().all()
    metadata_out = [i.todict() for i in db_entries]

    return metadata_out


@database_errors_handler
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

    fitted_model = mc.fit(X, y)
    metadata = get_model_metadata(fitted_model)

    new_db_entry = MlModel(
        model_class=metadata['model_class'],
        hyperparameters=metadata['hyperparameters'],
        model=metadata['model']
        )

    session.add(new_db_entry)
    session.commit()

    return {
        "status": "trained",
        "model_class": new_db_entry.model_class,
        "id": new_db_entry.id
        }


@database_errors_handler
@model_errors_handler
def predict_with_model(model_id, X):
    """
    Return predictions
    """
    db_entry = session.query(MlModel).filter(id=model_id).one()
    model = pickle.loads(db_entry.model)
    prediction = model.predict(X)

    if model.__class__.__name__ == "RandomForestClassifier":
        prediction = prediction.astype(str)

    return list(prediction)


@database_errors_handler
@model_errors_handler
def re_train(model_id, X, y):
    """
    Re-train an existing model
    """
    db_entry = session.query(MlModel).filter(id=model_id).one()
    model = pickle.loads(db_entry.model)

    refitted_model = model.fit(X, y)
    db_entry.model = pickle.dumps(refitted_model)
    session.add(db_entry)
    session.commit()

    return {"status": "re-trained", "id": db_entry.id}


@database_errors_handler
def delete_model(model_id):
    """
    Delete an existing model
    """
    db_entry = session.query(MlModel).filter(id=model_id).one()
    session.delete(db_entry) 
    session.commit()
