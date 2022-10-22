from lib2to3.pytree import Base
from flask import request
from flask_restx import Resource, Api
from flask_restx import fields
import itertools
from werkzeug.exceptions import BadRequest, NotFound
from . import auxiliary as aux


models = []
id_generator = itertools.count(start=1)
model_classes = aux.model_classes


api = Api(
    title="ML Models Api",
    version="1.0",
    description="MlOps course: Home Assignment #1 (Rest API)",
    contact="iaskvortsov@edu.hse.ru",
    doc="/ml_rest_api/doc",
)


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
        description="Records of training data",
        example=[{"c1": 1, "c2": 2}, {"c1": 3, "c2": 4}],
    ),
    "y": fields.Raw(description="Target values", example=[10, 11]),
}
train_fields = api.model("Train", train_fields)

wild = fields.Wildcard(fields.Raw)
re_train_fields = {
    "X": fields.Raw(
        description="Records of training data",
        example=[{"c1": 1, "c2": 2}, {"c1": 3, "c2": 4}],
    ),
    "y": fields.Raw(description="Target values", example=[10, 11]),
}
re_train_fields = api.model("Re-train", re_train_fields)

predict_fields = {
    "X": fields.Raw(
        description="Records of data", example=[{"c1": 1, "c2": 2}, {"c1": 3, "c2": 4}]
    )
}
predict_fields = api.model("Predict", predict_fields)


@api.route("/ml_rest_api/model_classes")
class GetModelClasses(Resource):
    def get(self):
        """
        Получить список доступных для обучения классов моделей
        """
        return model_classes


@api.route("/ml_rest_api/train/<string:model_class>")
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
            X = aux.prepare_X(request.get_json()["X"])
            y = aux.prepare_y(request.get_json()["y"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")

        fitted_model = aux.train_model(model_class, X, y, **hyperparameters)

        model_id = next(id_generator)
        models.append({"id": model_id, "model": fitted_model})

        return {"status": "trained", "model_class": model_class, "id": model_id}


@api.route("/ml_rest_api/retrain/<int:model_id>")
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
            X = aux.prepare_X(request.get_json()["X"])
            y = aux.prepare_y(request.get_json()["y"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")

        _ = aux.re_train(models, model_id, X, y)

        return {"status": "re-trained", "id": model_id}


@api.route("/ml_rest_api/saved_models")
class GetSavedModels(Resource):
    def get(self):
        """
        Вывести список имеющихся в памяти моделей
        """
        return {"models": aux.parse_models(models)}


@api.route("/ml_rest_api/predict/<int:model_id>")
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
            X = aux.prepare_X(request.get_json()["X"])
        except KeyError:
            raise BadRequest("Insufficient data provided.")
        prediction = aux.predict_with_model(models, model_id, X)
        return {"y_pred": prediction}


@api.route("/ml_rest_api/delete/<int:model_id>")
class DeleteModel(Resource):
    @api.doc(
        params={"model_id": "Id of a model used for prediction"}, responses=simple_resp
    )
    def delete(self, model_id):
        """
        Удалить обученную модель из памяти
        """
        global models
        if model_id not in [i["id"] for i in models]:
            e = NotFound("Model not found")
            raise e

        models = list(filter(lambda x: x["id"] != model_id, models))
        return {"status": "deleted", "model_id": model_id}
