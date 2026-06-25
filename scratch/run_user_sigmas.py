import sys
import os

# Append project root and scratch dir
project_root = r"c:\Users\mroemer\git-repositories\workshop_entscheidung_unsicherheit"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "scratch"))

import numpy as np
import pandas as pd
from src.costs import NewsvendorCosts
from run_experiments_no_spikes import run_backtest_experimental

# Parameters requested by user
# VK = 9.00, Material = 3.00, Restwert = 1.50
# co = 3.00 - 1.50 = 1.50
# cu = 9.00 - 3.00 = 6.00
co = 1.50
cu = 6.00

sigmas = [0.12, 0.20, 0.25, 0.30, 0.35, 0.40]

print("## ANNUAL PROFIT FOR CO=1.50, CU=6.00 (VK=9.00, Material=3.00, Restwert=1.50)")
print("| Sigma (Unsicherheit) | Mean-Based | XGBoost (Point) | NGBoost (Prob) | Gap (NGB vs XGB) | % Improvement |")
print("|---|---|---|---|---|---|")
for sigma in sigmas:
    res = run_backtest_experimental(
        n_years=8,
        seed=42,
        overage_cost=co,
        underage_cost=cu,
        sigma_base=sigma,
        include_spikes=False
    )
    gap = res["ngb_annual"] - res["xgb_annual"]
    pct = (gap / res["xgb_annual"]) * 100 if res["xgb_annual"] > 0 else 0
    print(f"| sigma = {sigma:.2f} | {res['mean_based_annual']:.2f} € | {res['xgb_annual']:.2f} € | {res['ngb_annual']:.2f} € | {gap:+.2f} € | {pct:.1f}% |")

print("\n## AVERAGE WEEKLY PROFIT FOR CO=1.50, CU=6.00 (VK=9.00, Material=3.00, Restwert=1.50)")
print("| Sigma (Unsicherheit) | Mean-Based | XGBoost (Point) | NGBoost (Prob) | Gap (NGB vs XGB) | % Improvement |")
print("|---|---|---|---|---|---|")
for sigma in sigmas:
    res = run_backtest_experimental(
        n_years=8,
        seed=42,
        overage_cost=co,
        underage_cost=cu,
        sigma_base=sigma,
        include_spikes=False
    )
    gap = res["ngb_weekly"] - res["xgb_weekly"]
    pct = (gap / res["xgb_weekly"]) * 100 if res["xgb_weekly"] > 0 else 0
    print(f"| sigma = {sigma:.2f} | {res['mean_based_weekly']:.2f} € | {res['xgb_weekly']:.2f} € | {res['ngb_weekly']:.2f} € | {gap:+.2f} € | {pct:.1f}% |")
