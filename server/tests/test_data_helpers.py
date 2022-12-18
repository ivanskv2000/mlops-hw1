import pytest
import json
from ml_rest_api import data_helpers
import numpy as np
import pandas as pd
import sklearn.datasets as skld


breast_cancer_data = skld.load_breast_cancer(as_frame=True)
iris_data = skld.load_iris(as_frame=True)
wine_data = skld.load_wine(as_frame=True)


def pandas_to_json(df):
    return json.loads(df.to_json(orient='split', index=False))


def json_to_pandas(js):
    return pd.DataFrame(js["data"], columns=js["columns"])


def df_equality(df1: pd.DataFrame, df2: pd.DataFrame):
    return np.all(df1.values == df2.values)


def list_equality(l1, l2):
    return np.all(np.array(l1) == np.array(l2))


class TestXPreparation:
    def test_breast_cancer(self):
        dh_result = data_helpers.prepare_X(
            pandas_to_json(
                breast_cancer_data['data']
                )
            )
        assert df_equality(dh_result, breast_cancer_data['data'])

    def test_iris(self):
        dh_result = data_helpers.prepare_X(
            pandas_to_json(
                iris_data['data']
                )
            )
        assert df_equality(dh_result, iris_data['data'])

    def test_wine(self):
        dh_result = data_helpers.prepare_X(
            pandas_to_json(
                wine_data['data']
                )
            )
        assert df_equality(dh_result, wine_data['data'])


class TestYPreparation:
    def test_breast_cancer(self):
        dh_result = data_helpers.prepare_y(
            breast_cancer_data['target'].tolist()
            )
        assert list_equality(dh_result, breast_cancer_data['target'].tolist())

    def test_iris(self):
        dh_result = data_helpers.prepare_y(
            iris_data['target'].tolist()
            )
        assert list_equality(dh_result, iris_data['target'].tolist())

    def test_wine(self):
        dh_result = data_helpers.prepare_y(
            wine_data['target'].tolist()
            )
        assert list_equality(dh_result, wine_data['target'].tolist())