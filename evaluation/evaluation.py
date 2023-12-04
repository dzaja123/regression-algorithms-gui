from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, r2_score
import pandas as pd
import math


def evaluate_metrics(Ytest: pd.DataFrame, real: pd.DataFrame) -> tuple:
    ex = explained_variance_score(Ytest, real)
    mse = mean_squared_error(Ytest, real)
    r2 = r2_score(Ytest, real)
    maxerr = max_error(Ytest, real)
    return ex, r2, mse, math.sqrt(mse), maxerr