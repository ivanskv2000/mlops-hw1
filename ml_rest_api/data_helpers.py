import pandas as pd
from werkzeug.exceptions import UnprocessableEntity


def prepare_X(X):
    """
    Convert json data to a Pandas dataframe
    """
    try:
        df = pd.DataFrame(X["data"], columns=X["columns"])
    except (TypeError, ValueError):
        raise UnprocessableEntity("Wrong data format provided.")

    return df


def prepare_y(y):
    """
    Convert json array to a Pandas dataframe
    """
    try:
        df = pd.Series(y).values
    except TypeError:
        raise UnprocessableEntity("Wrong data format provided.")

    return df
