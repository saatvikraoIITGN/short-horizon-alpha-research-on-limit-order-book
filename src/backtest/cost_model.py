"""
Transaction-cost model for short-horizon backtesting.

Costs are the primary reason short-horizon statistical alpha fails to
translate into tradable alpha.  We model three components:

1. **Exchange fee** (fixed bps per side)
2. **Spread cost** (multiplier x current half-spread)
3. **Slippage** (additional bps for market impact)

Four scenarios from the spec: zero, low, medium, high.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.paths import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def get_cost_scenarios(config: Optional[dict] = None) -> dict:
    """Return the cost-scenario dict from config."""
    if config is None:
        config = load_config()
    return config["backtest"]["cost_scenarios"]


def compute_trade_costs(
    spread: np.ndarray,
    mid_price: np.ndarray,
    scenario: str = "medium",
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute per-trade round-trip cost in return space.

    cost = 2 * fee_bps/1e4  +  spread_multiplier * spread / mid_price  +  2 * slippage_bps/1e4

    Returns an array of cost values (one per row), always positive.
    """
    scenarios = get_cost_scenarios(config)
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(scenarios.keys())}")

    s = scenarios[scenario]
    fee = 2 * s["fee_bps"] / 1e4
    slippage = 2 * s["slippage_bps"] / 1e4
    spread_cost = s["spread_multiplier"] * np.where(mid_price > 0, spread / mid_price, 0.0)

    total = fee + slippage + spread_cost
    return total


def apply_costs(
    gross_returns: np.ndarray,
    traded: np.ndarray,
    spread: np.ndarray,
    mid_price: np.ndarray,
    scenario: str = "medium",
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Subtract transaction costs from gross returns wherever a trade occurs.

    *traded* is a boolean or 0/1 array indicating when a new position is entered.
    """
    costs = compute_trade_costs(spread, mid_price, scenario, config)
    net = gross_returns - traded * costs
    return net
