# MlOps: HW-1

A simple restful API which is able to train sklearn models and get their predictions. Implements two model classes (LinearRegression, RandomForestClassifier).

## Usage

poetry run python3 app.py


## Methods
This API has a decent Swagger documentation available on `/ml_rest_api/doc`. 

___
### POST: train a model

**Parameters:**
- `model_class` &mdash; class of a model to train

**Payload:**
- `hyperparameters`
- `X` (predictors) and `y` (target). Both should be provided in a json records format (see example below).

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"hyperparameters": {}, "X": [{"c1":1, "c2":3}, {"c1":0, "c2":13}, {"c1":-3, "c2":3.5}], "y": [1,2,3]}' http://127.0.0.1:5000/ml_rest_api/train/LinearRegression
```

___
### GET: return model classes which can be trained

**Example:**
```bash
curl http://127.0.0.1:5000/ml_rest_api/model_classes
```

___
### GET: return the list of saved models

**Example:**
```
curl http://127.0.0.1:5000/ml_rest_api/saved_models
```

___
### POST: predict with a particular model

**Parameters:**
- `model_id` &mdash; id of a model to use for prediction

**Payload:**
- `X` (predictors)

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"X": [{"c1":1, "c2":3}, {"c1":0, "c2":13}, {"c1":-3, "c2":3.5}]}' http://127.0.0.1:5000/ml_rest_api/predict/1
```

___
### PUT: re-train an existing model

**Parameters:**
- `model_id` &mdash; id of a model to re-train

**Payload:**
- `X` (predictors) and `y` (target)

**Example:**
```bash
curl -X PUT -H "Content-Type: application/json" http://127.0.0.1:5000/ml_rest_api/retrain/1 -d '{"X": [{"c1":345, "c2":3222}, {"c1":134, "c2":1003}, {"c1":215, "c2":999}], "y": [10000,23335,34556]}'
```

___
### DELETE: delete an existing model

**Parameters:**
- `model_id` &mdash; id of a model to delete

**Example:**
```bash
curl -X DELETE http://127.0.0.1:5000/ml_rest_api/delete/1
```