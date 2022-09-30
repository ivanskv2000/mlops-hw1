#from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from flask import request
from flask_restx import Resource, Api
from flask_restx import fields
import itertools
from werkzeug.exceptions import BadRequest
from . import auxiliary as aux


models = []
id_generator = itertools.count(start=1)
model_classes = {
    'classes': [
        {
            'name': 'LinearRegression',
            'predictors': ['numeric'],
            'target': ['numeric'],
            # 'hyperparameters': ['fit_intercept']
        },
        {
            'name': 'RandomForestClassifier',
            'predictors': ['numeric', 'categorical'],
            'target': ['categorical'],
            # 'hyperparameters': ['n_estimators', 'criterion', 'max_depth']
        }
    ]
}

api = Api(
    title="ML Models Api",
    version="1.0",
    description="Home Assignment #1 for MlOps course",
    contact="iaskvortsov@edu.hse.ru",
    doc="/ml_rest_api/doc"
    )



@api.route('/ml_rest_api')
class GeneralInfo(Resource):
    def get(self):
        return {
            'api': api.title,
            'version': api.version,
            'contact': api.contact
            }


@api.route('/ml_rest_api/model_classes')
class GetModelClasses(Resource):
    def get(self):
        return model_classes


wild = fields.Wildcard(fields.Raw)
train_fields = {
    'hyperparameters': wild,
    'X': fields.Raw(
        description='Records of training data',
        example=[{"c1": 1, "c2": 2}, {"c1": 3, "c2": 4}]
        ),
    'y': fields.Raw(
        description='Target values',
        example=[10, 11]
        )
}
train_fields = api.model('Train', train_fields)


@api.route('/ml_rest_api/train/<string:model_class>')
class TrainModel(Resource):
    @api.expect(train_fields)
    @api.doc(params={'model_class': 'The class of a model to train'})
    def post(self, model_class):
        if model_class == 'LinearRegression':
            mc = LinearRegression(**request.get_json()['hyperparameters'])
        elif model_class == 'RandomForestClassifier':
            mc = RandomForestClassifier(**request.get_json()['hyperparameters'])
        else:
            raise BadRequest('Unknown model class provided')

        print('Fitting model...')
        X = aux.prepare_X(request.get_json()['X'])
        y = aux.prepare_y(request.get_json()['y'])
        fitted_model = mc.fit(X, y)
        model_id = next(id_generator)
        models.append({'id': model_id, 'model': fitted_model})

        return {'model_class': model_class, 'id': model_id}


@api.route('/ml_rest_api/saved_models')
class GetSavedModels(Resource):
    def get(self):
        return {'models': aux.parse_models(models)}


predict_fields = {
    'X': fields.Raw(
        description='Records of data', 
        example=[{"c1": 1, "c2": 2}, {"c1": 3, "c2": 4}]
        )
}
predict_fields = api.model('Predict', predict_fields)


@api.route('/ml_rest_api/predict/<int:model_id>')
class PredictWithExisting(Resource):
    @api.expect(predict_fields)
    @api.doc(params={'model_id': 'Id of a model used for prediction'})
    def post(self, model_id):
        X = aux.prepare_X(request.get_json()['X'])
        model = list(filter(lambda x: x['id'] == model_id, models))[0]['model']
        prediction = model.predict(X)

        return list(prediction)


@api.route('/ml_rest_api/delete/<int:model_id>')
class DeleteModel(Resource):
    @api.doc(params={'model_id': 'Id of a model used for prediction'})
    def get(self, model_id):
        global models
        models = list(filter(lambda x: x['id'] != model_id, models))
        return model_id
