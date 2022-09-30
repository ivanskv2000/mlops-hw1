# MlOps: HW-1

http://[hostname]/ml_rest_api/
poetry run python3 app.py

Методы

POST: обучение ML-модели
- тип модели
- гиперпараметры
- данные
curl -X POST -H "Content-Type: application/json" -d '{"hyperparameters": {}, "X": [{"c1":1, "c2":3}, {"c1":0, "c2":13}, {"c1":-3, "c2":3.5}], "y": [1,2,3]}' http://127.0.0.1:5000/ml_rest_api/train/LinearRegression

GET: список доступных для обучения классов моделей
curl http://127.0.0.1:5000/ml_rest_api/model_classes

GET: список имеющихся в памяти моделей
curl http://127.0.0.1:5000/ml_rest_api/saved_models

GET: предсказание конкретной модели
- id модели
- данные
curl -X POST -H "Content-Type: application/json" -d '{"X": [{"c1":1, "c2":3}, {"c1":0, "c2":13}, {"c1":-3, "c2":3.5}]}' http://127.0.0.1:5000/ml_rest_api/predict/1

POST: обучить модель заново
- id модели
- гиперпараметры
- данные 

GET: удалить обученную модель
- id модели
curl http://127.0.0.1:5000/ml_rest_api/delete/1


X - подаем в виде [{'col1': zzz, 'col2': zzz}, {'col1': yyy, 'col2': yyy}, ...]
y - подаем в виде [x, y, z, ...]