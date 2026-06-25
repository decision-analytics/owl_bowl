"""
Module for data generation and splitting.
"""

import numpy as np
import pandas as pd

FEATURES = ["week_sin", "week_cos", "temp", "holiday", "rain", "event"]

def simulate_history(n_years: int = 8, seed: int = 42, include_spikes: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for year in range(1, n_years + 1):
        for week in range(1, 53):
            theta = 2 * np.pi * (week - 1) / 52.0

            # Beobachtbare Kontextinformationen
            temp = 11 + 12 * np.sin(theta - 0.9) + rng.normal(0, 3.0)
            holiday = int((week in [1, 2, 31, 32, 33, 52]))
            rain = rng.binomial(1, 0.35 - 0.10 * np.sin(theta - 0.7))

            # Lokales Event: im Sommer etwas wahrscheinlicher
            p_event = 0.04 + 0.10 * max(0.0, np.sin(theta - 0.2))
            event = rng.binomial(1, min(max(p_event, 0.0), 0.5))

            # Struktureller Teil der Nachfrage
            log_location = (
                4.15
                + 0.12 * np.sin(theta - 0.5)
                + 0.08 * np.cos(theta)
                + 0.015 * temp
                + 0.18 * event
                - 0.14 * holiday
                - 0.08 * rain
            )

            # Aleatorische Restunsicherheit
            sigma = 0.25 + 0.03 * (temp > 20) + 0.05 * event
            eps = rng.normal(0.0, sigma)

            # Seltene positive Nachfragespitzen
            if include_spikes:
                spike_prob = 0.03 + 0.05 * event + 0.02 * (temp > 24)
                spike = rng.exponential(scale=0.55) if rng.random() < spike_prob else 0.0
            else:
                spike = 0.0

            log_demand = log_location + eps + spike
            demand = int(np.round(np.exp(log_demand)))
            demand = max(demand, 0)

            rows.append(
                {
                    "year": year,
                    "week": week,
                    "week_sin": np.sin(theta),
                    "week_cos": np.cos(theta),
                    "temp": temp,
                    "holiday": holiday,
                    "rain": rain,
                    "event": event,
                    "demand": demand,
                }
            )

    return pd.DataFrame(rows)


def split_train_test(df: pd.DataFrame, n_test_years: int = 1):
    max_year = df["year"].max()
    train = df[df["year"] <= max_year - n_test_years].copy()
    test = df[df["year"] > max_year - n_test_years].copy()
    return train, test


def simulate_true_demand_for_week_context(row: pd.Series, n_samples: int = 10000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    week = row["week"]
    theta = 2 * np.pi * (week - 1) / 52.0
    temp = row["temp"]
    event = row["event"]
    holiday = row["holiday"]
    rain = row["rain"]

    log_location = (
        4.15
        + 0.12 * np.sin(theta - 0.5)
        + 0.08 * np.cos(theta)
        + 0.015 * temp
        + 0.18 * event
        - 0.14 * holiday
        - 0.08 * rain
    )

    sigma = 0.25 + 0.03 * (temp > 20) + 0.05 * event
    spike_prob = 0.03 + 0.05 * event + 0.02 * (temp > 24)

    eps = rng.normal(0.0, sigma, size=n_samples)
    spikes = np.where(rng.random(size=n_samples) < spike_prob, rng.exponential(scale=0.55, size=n_samples), 0.0)

    log_demand = log_location + eps + spikes
    demand = np.maximum(0, np.rint(np.exp(log_demand))).astype(int)
    
    return demand
