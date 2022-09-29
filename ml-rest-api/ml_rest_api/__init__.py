#from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from flask import request
from flask_restx import Resource, Api
import itertools
from werkzeug.exceptions import BadRequest
#import joblib

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


@api.route('/ml_rest_api/train/<string:model_class>')
class TrainModel(Resource):
    def post(self, model_class):
        if model_class == 'LinearRegression':
            mc = LinearRegression(**request.json['hyperparameters'])
        elif model_class == 'RandomForestClassifier':
            mc = RandomForestClassifier(**request.json['hyperparameters'])
        else:
            raise BadRequest('Unknown model class provided')

        print('Fitting model...')
        fitted_model = mc.fit(request.json['X'], request.json['y'])
        model_id = next(id_generator)
        models.append({'id': model_id, 'model': fitted_model})

        return {'model_class': model_class, 'id': model_id}


@api.route('/ml_rest_api/saved_models')
class GetSavedModels(Resource):
    def parse_models(self, models_list):
        out = [{'id': m['id'], 'model_class': m['model'].__class__.__name__} for m in models_list]

        return out

    def get(self):
        return {'models': self.parse_models(models)}


@api.route('/ml_rest_api/predict/<int:model_id>')
class PredictWithExisting(Resource):
    def get(self, model_id):
        predictors = request.json['X']
        model = filter(lambda x: x['id'] == model_id, models)
        prediction = model.predict(predictors)

        return prediction
