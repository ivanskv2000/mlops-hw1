# MlOps: HW-1

A simple restful API which is able to train sklearn models and get their predictions. Implements two model classes (LinearRegression, RandomForestClassifier).

## Usage

### Step 1. Set Up the Working Environment

> :warning: I used Poetry to manage package dependencies and other configurations. If you don't have Poetry on your machine, refer to their [documentation](https://python-poetry.org/docs/).

Initialise the project in Poetry
```bash
cd path_to_project/
poetry init
```

Install required dependencies
```bash
poetry install
```

Run server
```bash
poetry run python3 app.py
```

All done! Now you can visit API's main page on [127.0.0.1:5000/ml_rest_api](http://127.0.0.1:5000/ml_rest_api).

## Data Format
In cases when you should pass the data (X and/or y, depending on the method), you should follow the **records** format. For `X` (predictor) data, it is an array of observations with each observation presented as a dictionary:

```
[{"col1":<val>, "col2":<val>, "col3":<val>, ...}, {"col1":<val>, "col2": <val>, "col3":<val>, ...}, ...]
```

For `y` (target) data it is just a flat array:

```
[<val>, <val>, ...]
```

## Methods
This API has a decent Swagger documentation available on [`/ml_rest_api/doc`](http://127.0.0.1:5000/ml_rest_api/doc). 

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