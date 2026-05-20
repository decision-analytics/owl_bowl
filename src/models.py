"""
Module for forecasting models.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

from src.data import FEATURES

def fit_mean_based_mean(train: pd.DataFrame) -> float:
    return float(train["demand"].mean())

def predict_mean_based_mean(mean_value: float, n: int) -> np.ndarray:
    return np.full(n, mean_value, dtype=float)

def fit_xgb_point_model(train: pd.DataFrame):
    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=0,
    )
    model.fit(train[FEATURES], train["demand"])
    return model

def fit_ngboost_model(train: pd.DataFrame):
    model = NGBRegressor(
        Dist=Normal,
        n_estimators=300,
        learning_rate=0.03,
        natural_gradient=True,
        random_state=0,
        verbose=False,
    )
    model.fit(train[FEATURES].to_numpy(), train["demand"].to_numpy())
    return model

def sample_ngboost_predictive_distribution(
    ngb_model,
    X_row,
    n_samples: int = 1500,
) -> np.ndarray:
    pred_dist = ngb_model.pred_dist(X_row)
    raw = pred_dist.sample(n_samples)
    samples = np.asarray(raw).reshape(-1)
    samples = np.maximum(0, samples)
    return samples
