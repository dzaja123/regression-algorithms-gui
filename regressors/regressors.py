from evaluation.evaluation import evaluate_metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np


def multiple_linear_regression(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame) -> tuple:
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    real_mlreg = model.predict(Xtest)
    metrics = evaluate_metrics(Ytest, real_mlreg)
    return metrics

def polynomial_regression(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame) -> tuple:
    poly = PolynomialFeatures(degree=2)
    polyTrainX = poly.fit_transform(Xtrain)
    polyTestX = poly.fit_transform(Xtest)
    pol = LinearRegression()
    pol.fit(polyTrainX, Ytrain)
    real_pol = pol.predict(polyTestX)
    metrics = evaluate_metrics(Ytest, real_pol)
    return metrics

def decision_tree_regression(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame) -> tuple:
    tree = DecisionTreeRegressor()
    tree.fit(Xtrain, Ytrain)
    real_tree = tree.predict(Xtest)
    metrics = evaluate_metrics(Ytest, real_tree)
    return metrics

def random_forest_regression(Xtrain: pd.DataFrame, rYtrain: np.ndarray, Xtest: pd.DataFrame, rYtest: np.ndarray) -> tuple:
    forest = RandomForestRegressor()
    forest.fit(Xtrain, rYtrain)
    real_forest = forest.predict(Xtest)
    metrics = evaluate_metrics(rYtest, real_forest)
    return metrics

def support_vector_machine(Xtrain: pd.DataFrame, rYtrain: np.ndarray, Xtest: pd.DataFrame, rYtest: np.ndarray) -> tuple:
    svr = SVR(kernel="poly", degree=2, C=0.5, epsilon=5000)
    svr.fit(Xtrain, rYtrain)
    real_svr = svr.predict(Xtest)
    metrics = evaluate_metrics(rYtest, real_svr)
    return metrics
