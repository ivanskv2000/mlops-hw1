from flask import request
from flask_restx import Namespace, Resource
from flask_restx import fields
import itertools
from werkzeug.exceptions import BadRequest, NotFound
from . import data_helpers
from . import model_helpers
import os
from .db_models import session


models_metadata = []

proj_path = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
models_path = os.path.join(proj_path, "models")

ids = [
    1,
]
for file in os.listdir(models_path):
    if file.endswith(".txt"):
        id_curr = int(file.split(".")[0])
        ids.append(id_curr)
max_id = max(ids)

id_generator = itertools.count(start=max_id)
model_classes = model_helpers.model_classes

api = Namespace("ml_rest_api")


@api.errorhandler(NotFound)
def handle_no_result_exception(error):
    """Return a model not found error message and 404 status code"""
    return {
        "message": "Model not found. Check available models using /ml_rest_api/saved_models"
    }, 404


model_resp = {
    200: "Success",
    400: "Bad Request",
    404: "Not Found",
    422: "Unprocessable Entity",
}

simple_resp = {
    200: "Success",
    404: "Not Found",
}


wild = fields.Wildcard(fields.Raw)
train_fields = {
    "hyperparameters": wild,
    "X": fields.Raw(
        description="Training data",
        example={
            "columns": ["one", "two", "three"],
            "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
    ),
    "y": fields.Raw(description="Target values", example=[10, 11]),
}
train_fields = api.model("Train", train_fields)

wild = fields.Wildcard(fields.Raw)
re_train_fields = {
    "X": fields.Raw(
        description="Records of training data",
        example={"columns": ["c1", "c2"], "data": [[1, 3.0], [0, 13.0], [-3, 3.5]]},
    ),
    "y": fields.Raw(description="Target values", example=[1, 2, 3]),
}
re_train_fields = api.model("Re-train", re_train_fields)

predict_fields = {
    "X": fields.Raw(
        description="Records of data",
        example={"columns": ["c1", "c2"], "data": [[1, 2], [3, 4]]},
    )
}
predict_fields = api.model("Predict", predict_fields)


@api.route("/model_classes")
class GetModelClasses(Resource):
    def get(self):
        """
        Получить список доступных для обучения классов моделей
        """
        return model_classes


@api.route("/train/<string:model_class>")
class TrainModel(Resource):
    @api.expect(train_fields)
    @api.doc(
        params={"model_class": "The class of a model to train"}, responses=model_resp
    )
    def post(self, model_class):
        """
        Обучить ML-модель и сохранить ее в памяти
        """
        hyperparameters = request.get_json().get("hyperparameters", {})
        try:
            X = data_helpers.prepare_X(request.get_json()["X"])
            y = data_helpers.prepare_y(request.get_json()["y"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")

        return model_helpers.train_model(model_class, X, y, **hyperparameters)


@api.route("/retrain/<int:model_id>")
class ReTrainModel(Resource):
    @api.expect(re_train_fields)
    @api.doc(
        params={"model_id": "Id of a model used for prediction"}, responses=model_resp
    )
    def put(self, model_id):
        """
        Обучить сохраненную модель заново
        """
        try:
            X = data_helpers.prepare_X(request.get_json()["X"])
            y = data_helpers.prepare_y(request.get_json()["y"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")

        return model_helpers.re_train(model_id, X, y)


@api.route("/saved_models")
class GetSavedModels(Resource):
    def get(self):
        """
        Вывести список имеющихся в памяти моделей
        """
        return {"models": model_helpers.parse_models()}


@api.route("/predict/<int:model_id>")
class PredictWithExisting(Resource):
    @api.expect(predict_fields)
    @api.doc(
        params={"model_id": "Id of a model used for prediction"}, responses=model_resp
    )
    def post(self, model_id):
        """
        Получить предсказание выбранной модели
        """
        try:
            X = data_helpers.prepare_X(request.get_json()["X"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")
        prediction = model_helpers.predict_with_model(model_id, X)
        return {"y_pred": prediction}


@api.route("/delete/<int:model_id>")
class DeleteModel(Resource):
    @api.doc(
        params={"model_id": "Id of a model used for prediction"}, responses=simple_resp
    )
    def delete(self, model_id):
        """
        Удалить обученную модель из памяти
        """
        return model_helpers.delete_model(model_id)

