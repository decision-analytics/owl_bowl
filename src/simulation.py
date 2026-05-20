"""
Module for running the backtest simulation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import FEATURES, split_train_test, simulate_history
from src.costs import NewsvendorCosts, newsvendor_cost
from src.models import (
    fit_mean_based_mean,
    predict_mean_based_mean,
    fit_xgb_point_model,
    fit_ngboost_model,
    sample_ngboost_predictive_distribution,
)
from src.decision import clip_round_positive, choose_quantity_by_quantile_grid


def run_backtest_one_world(
    n_years: int = 8,
    seed: int = 42,
    overage_cost: float = 1.5,
    underage_cost: float = 9.0,
    n_samples_ngb: int = 1500,
    tau_low: float = 0.50,
    tau_high: float = 0.99,
    tau_steps: int = 40,
    include_spikes: bool = True,
    n_test_years: int = 1,
):
    costs = NewsvendorCosts(
        overage_cost=overage_cost,
        underage_cost=underage_cost,
    )

    df = simulate_history(n_years=n_years, seed=seed, include_spikes=include_spikes)
    train, test = split_train_test(df, n_test_years=n_test_years)

    X_test = test[FEATURES]
    y_test = test["demand"].to_numpy()

    tau_grid = np.linspace(tau_low, tau_high, tau_steps)

    # Phase 1: Mean-based
    mean_based_mean = fit_mean_based_mean(train)
    forecast_mean_based = predict_mean_based_mean(mean_based_mean, len(test))

    # Phase 2: XGBoost Punktprognose
    xgb_model = fit_xgb_point_model(train)
    forecast_xgb = xgb_model.predict(X_test)

    # Forecasting-Metriken
    mae_mean_based = mean_absolute_error(y_test, forecast_mean_based)
    mae_xgb = mean_absolute_error(y_test, forecast_xgb)

    rmse_mean_based = np.sqrt(mean_squared_error(y_test, forecast_mean_based))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, forecast_xgb))

    # Phase 3: Eingebettete Policy q = Punktforecast
    q_mean_based = clip_round_positive(forecast_mean_based)
    q_xgb = clip_round_positive(forecast_xgb)

    cost_mean_based = newsvendor_cost(q_mean_based, y_test, costs)
    cost_xgb = newsvendor_cost(q_xgb, y_test, costs)

    # Phase 4: NGBoost probabilistisch + brute-force Quantilgitter (vektorisiert für maximale Performance)
    ngb_model = fit_ngboost_model(train)

    # 1. Batch-Vorhersage und Sampling für alle Test-Wochen auf einmal
    pred_dist = ngb_model.pred_dist(X_test.to_numpy())
    raw_samples = pred_dist.sample(n_samples_ngb)  # Shape: (n_samples_ngb, W)
    samples_matrix = np.maximum(0, raw_samples)    # Shape: (n_samples_ngb, W)
    W = len(test)

    # 2. Vektorisierte Quantil-Berechnung über alle Wochen und Quantile (M, W)
    q_taus = np.round(np.quantile(samples_matrix, tau_grid, axis=0)).astype(int)

    # 3. Vektorisierte Kosten-Berechnung per 3D Broadcasting
    # q_taus_3d Shape: (M, 1, W)
    # samples_3d Shape: (1, n_samples_ngb, W)
    q_taus_3d = q_taus[:, np.newaxis, :]
    samples_3d = samples_matrix[np.newaxis, :, :]
    
    over = np.maximum(q_taus_3d - samples_3d, 0)
    under = np.maximum(samples_3d - q_taus_3d, 0)
    cost_matrix = costs.overage_cost * over + costs.underage_cost * under  # Shape: (M, n_samples_ngb, W)
    avg_costs = cost_matrix.mean(axis=1)  # Shape: (M, W)

    # 4. Beste Entscheidung für jede Woche ermitteln
    best_indices = np.argmin(avg_costs, axis=0)  # Shape: (W,)
    
    q_ngb = q_taus[best_indices, np.arange(W)]
    tau_ngb = tau_grid[best_indices]
    expected_cost_ngb = avg_costs[best_indices, np.arange(W)]
    demand_mean_ngb = samples_matrix.mean(axis=0)

    # 5. Beispiel-Scantabelle für die erste Woche (w = 0) konstruieren
    q_scan_example = pd.DataFrame({
        "tau": tau_grid,
        "q": q_taus[:, 0],
        "avg_cost": avg_costs[:, 0]
    }).sort_values(["avg_cost", "tau"], ascending=[True, True]).reset_index(drop=True)

    cost_ngb = newsvendor_cost(q_ngb, y_test, costs)

    detail = test.copy()
    detail["forecast_mean_based"] = forecast_mean_based
    detail["forecast_xgb"] = forecast_xgb
    detail["forecast_ngb_mean"] = demand_mean_ngb
    detail["q_mean_based"] = q_mean_based
    detail["q_xgb"] = q_xgb
    detail["q_ngb"] = q_ngb
    detail["tau_ngb"] = tau_ngb
    detail["expected_cost_ngb"] = expected_cost_ngb
    detail["cost_mean_based"] = cost_mean_based
    detail["cost_xgb"] = cost_xgb
    detail["cost_ngb"] = cost_ngb
    detail["cum_cost_mean_based"] = detail["cost_mean_based"].cumsum()
    detail["cum_cost_xgb"] = detail["cost_xgb"].cumsum()
    detail["cum_cost_ngb"] = detail["cost_ngb"].cumsum()

    q_scan_all_weeks = pd.DataFrame({
        "tau": tau_grid,
        "avg_cost": avg_costs.mean(axis=1)
    }).sort_values("tau").reset_index(drop=True)

    summary = pd.Series(
        {
            "seed": seed,
            "n_years": n_years,
            "critical_fractile_reference": costs.critical_fractile,
            "mean_based_mean": float(mean_based_mean),
            "mae_mean_based": float(mae_mean_based),
            "mae_xgb": float(mae_xgb),
            "rmse_mean_based": float(rmse_mean_based),
            "rmse_xgb": float(rmse_xgb),
            "annual_cost_mean_based_point": float(cost_mean_based.sum() / n_test_years),
            "annual_cost_xgb_point": float(cost_xgb.sum() / n_test_years),
            "annual_cost_ngb_prob": float(cost_ngb.sum() / n_test_years),
            "avg_weekly_cost_mean_based_point": float(cost_mean_based.mean()),
            "avg_weekly_cost_xgb_point": float(cost_xgb.mean()),
            "avg_weekly_cost_ngb_prob": float(cost_ngb.mean()),
        }
    )

    return summary, detail, q_scan_example, df, q_scan_all_weeks
