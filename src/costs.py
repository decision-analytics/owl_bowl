"""
Module for cost definitions.
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class NewsvendorCosts:
    overage_cost: float = 0.5
    underage_cost: float = 7.0

    @property
    def critical_fractile(self) -> float:
        return self.underage_cost / (self.underage_cost + self.overage_cost)

def newsvendor_cost(q, d, costs: NewsvendorCosts):
    q = np.asarray(q)
    d = np.asarray(d)
    over = np.maximum(q - d, 0)
    under = np.maximum(d - q, 0)
    return costs.overage_cost * over + costs.underage_cost * under

def newsvendor_profit(q, d, costs: NewsvendorCosts):
    q = np.asarray(q)
    d = np.asarray(d)
    return costs.underage_cost * d - newsvendor_cost(q, d, costs)

