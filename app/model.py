from sklearn.linear_model import LogisticRegression
import os
from pathlib import Path
import joblib
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score


def train_load(model_path='models/loan.joblib'):
    if Path(model_path).exists():
        return joblib.load(model_path)
    df = pd.read_csv('data/loan200.csv')
    train = df.iloc[1:]
    test = df.iloc[:1]
    X, y = train.drop('outcome', axis=1), (train['outcome']=='default').astype(int)
    model = LogisticRegression().fit(X, y)
    print("Accuracy score: ", accuracy_score(y, model.predict(X)))
    joblib.dump(model, model_path)
    return model