from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def prepare_dataset(df: pd.DataFrame) -> tuple:
    df = encode_labels(df)
    df = impute_missing_values(df)
    X = df[["longitude", 
            "latitude", 
            "housing_median_age", 
            "total_rooms", 
            "total_bedrooms", 
            "population", 
            "households", 
            "median_income", 
            "ocean_proximity"]]
    Y = df[["median_house_value"]]

    Xtrain, Xtest, Ytrain, Ytest = split_data(X, Y)
    Xtrain, Xtest = scale_data(Xtrain, Xtest)
    rYtrain, rYtest = ravel_transform(Ytrain, Ytest)
    return Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    df["ocean_proximity"] = label_encoder.fit_transform(df["ocean_proximity"])
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='mean')
    df["total_bedrooms"] = imputer.fit_transform(df[["total_bedrooms"]])
    return df

def split_data(X, Y, test_size: float = 0.2, random_state: int = 42) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def ravel_transform(Ytrain: np.ndarray, Ytest: np.ndarray) -> tuple:
    rYtrain = np.ravel(Ytrain)
    rYtest = np.ravel(Ytest)
    return rYtrain, rYtest