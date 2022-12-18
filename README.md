# MlOps: HW-1

A simple restful API which is able to train sklearn models and get their predictions. Implements two model classes (LinearRegression, RandomForestClassifier).

Full API is a `docker-compose` application with two services: API itself and a PostrgeSQL database used to store models and their metadata.

## Usage

### Step 1. Get Docker

If you don't have Docker installed on your machine, refer to their [documentation](https://docs.docker.com/get-docker/).

### Step 2. Build `docker-compose` application.

```bash
cd path_to_project/
docker-compose up -d
```

All done! Now you can visit API's main page on [127.0.0.1:8000/ml_rest_api](http://127.0.0.1:8000/ml_rest_api).

## Data Format
In cases when you pass the data (X and/or y, depending on the method), you should follow the specific format. For `X` (predictor) data, it is necessary to specify column names and the data itselt as a 2-D array:

```
{"columns": ["col1","col2"], "data": [[<val1>, <val2>],[<val3>, <val4>],[<val5>, <val6>], ...]}
```

For `y` (target) data it is just a flat array:

```
[<val>, <val>, ...]
```

## Methods
This API has a decent Swagger documentation available on [`/ml_rest_api/doc`](http://127.0.0.1:8000/ml_rest_api/doc). 

___
### POST: train a model

**Parameters:**
- `model_class` &mdash; class of a model to train (`LinearRegression` or `RandomForestClassifier`)

**Payload:**
- `hyperparameters`
- `X` (predictors) and `y` (target). Both should be provided in a json records format (see example below).

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"hyperparameters": {}, "X": {"columns": ["c1","c2"], "data": [[1,3.0],[0,13.0],[-3,3.5]]}, "y": [1,2,3]}' http://127.0.0.1:8000/ml_rest_api/train/LinearRegression
```

___
### GET: return model classes which can be trained

**Example:**
```bash
curl http://127.0.0.1:8000/ml_rest_api/model_classes
```

___
### GET: return the list of saved models

**Example:**
```
curl http://127.0.0.1:8000/ml_rest_api/saved_models
```

___
### POST: predict with a particular model

**Parameters:**
- `model_id` &mdash; id of a model to use for prediction

**Payload:**
- `X` (predictors)

**Example:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"X": {"columns": ["c1","c2"], "data": [[1,3.0],[0,13.0],[-3,3.5]]}}' http://127.0.0.1:8000/ml_rest_api/predict/1
```

___
### PUT: re-train an existing model

**Parameters:**
- `model_id` &mdash; id of a model to re-train

**Payload:**
- `X` (predictors) and `y` (target)

**Example:**
```bash
curl -X PUT -H "Content-Type: application/json" http://127.0.0.1:8000/ml_rest_api/retrain/1 -d '{"X": {"columns": ["c1","c2"], "data": [[345,3222],[134,1003],[215,999]]}, "y": [10000,23335,34556]}'
```

___
### DELETE: delete an existing model

**Parameters:**
- `model_id` &mdash; id of a model to delete

**Example:**
```bash
curl -X DELETE http://127.0.0.1:8000/ml_rest_api/delete/1
```

## Testing

To `pytest` the api, run 

```bash
cd server/
poetry run python -m pytest
```