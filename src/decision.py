"""
Module for decision rules.
"""
import numpy as np
import pandas as pd

from src.costs import NewsvendorCosts, newsvendor_cost

def clip_round_positive(x):
    return np.maximum(0, np.rint(x)).astype(int)

def choose_quantity_by_quantile_grid(
    demand_samples: np.ndarray,
    costs: NewsvendorCosts,
    tau_grid: np.ndarray | None = None,
):
    demand_samples = np.asarray(demand_samples)

    if tau_grid is None:
        tau_grid = np.linspace(0.50, 0.99, 40)

    # Vectorized quantile computation
    q_taus = np.round(np.quantile(demand_samples, q=tau_grid)).astype(int)

    # Vectorized cost and profit computation
    q_taus_2d = q_taus[:, np.newaxis]
    demand_2d = demand_samples[np.newaxis, :]
    over = np.maximum(q_taus_2d - demand_2d, 0)
    under = np.maximum(demand_2d - q_taus_2d, 0)
    costs_matrix = costs.overage_cost * over + costs.underage_cost * under
    profit_matrix = costs.underage_cost * demand_2d - costs_matrix
    avg_profits = profit_matrix.mean(axis=1)

    rows = []
    for i, tau in enumerate(tau_grid):
        rows.append(
            {
                "tau": float(tau),
                "q": int(q_taus[i]),
                "avg_profit": float(avg_profits[i]),
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(["avg_profit", "tau"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best = table.iloc[0].to_dict()
    return best, table

