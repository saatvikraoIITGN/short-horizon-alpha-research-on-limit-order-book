"""
Signal-based long/short backtest.

Trading rule (from spec):
- Go **long** when predicted return is in the top decile.
- Go **short** when predicted return is in the bottom decile.
- Stay **flat** otherwise.
- Hold for h seconds (the prediction horizon).
- Non-overlapping trades for cleaner analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.cost_model import compute_trade_costs
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_positions(
    predictions: pd.Series,
    long_quantile: float = 0.9,
    short_quantile: float = 0.1,
) -> pd.Series:
    """
    Map continuous predictions to discrete positions: +1 / 0 / -1.

    Thresholds are computed on the full prediction series (in-sample
    quantiles on the test set -- standard for signal-based research).
    """
    q_long = predictions.quantile(long_quantile)
    q_short = predictions.quantile(short_quantile)

    positions = pd.Series(0, index=predictions.index, dtype=np.int8)
    positions[predictions >= q_long] = 1
    positions[predictions <= q_short] = -1

    logger.info(
        "Positions: long=%d  short=%d  flat=%d  (thresholds: %.6f / %.6f)",
        (positions == 1).sum(), (positions == -1).sum(), (positions == 0).sum(),
        q_long, q_short,
    )
    return positions


def non_overlapping_positions(
    positions: pd.Series,
    hold_periods: int = 5,
) -> pd.Series:
    """
    Enforce non-overlapping trades: once a position is entered, hold for
    *hold_periods* rows, then go flat until the next signal.
    """
    result = pd.Series(0, index=positions.index, dtype=np.int8)
    i = 0
    idx = positions.index.tolist()

    while i < len(idx):
        pos = positions.iloc[i]
        if pos != 0:
            for j in range(hold_periods):
                if i + j < len(idx):
                    result.iloc[i + j] = pos
            i += hold_periods
        else:
            i += 1

    return result


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    horizon: str = "5s",
    frequency: str = "1s",
    long_quantile: float = 0.9,
    short_quantile: float = 0.1,
    cost_scenarios: list = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Run a complete signal-based backtest.

    Returns a dict with positions, gross/net returns per cost scenario,
    and summary statistics.
    """
    if cost_scenarios is None:
        cost_scenarios = ["zero", "low", "medium", "high"]

    hold_periods = int(pd.Timedelta(horizon) / pd.Timedelta(frequency))
    return_col = f"y_return_{horizon}"

    # Align predictions to the dataframe
    common = df.index.intersection(predictions.index)
    pred_aligned = predictions.loc[common]
    df_aligned = df.loc[common].copy()

    if return_col not in df_aligned.columns:
        raise ValueError(f"Target column {return_col} not found")

    # Generate positions
    raw_pos = generate_positions(pred_aligned, long_quantile, short_quantile)
    positions = non_overlapping_positions(raw_pos, hold_periods)
    df_aligned["position"] = positions

    # Gross returns
    actual_return = df_aligned[return_col].values
    pos_arr = positions.values.astype(float)
    gross = pos_arr * actual_return
    df_aligned["gross_return"] = gross

    # Track when trades happen (position changes)
    traded = np.abs(np.diff(pos_arr, prepend=0)) > 0
    df_aligned["traded"] = traded.astype(np.int8)

    spread = df_aligned["spread"].values
    mid = df_aligned["mid_price"].values

    # Net returns per cost scenario
    net_returns = {}
    for sc in cost_scenarios:
        costs = compute_trade_costs(spread, mid, sc, config)
        net = gross - traded * costs
        col = f"net_return_{sc}"
        df_aligned[col] = net
        net_returns[sc] = net

    # Summary
    n_trades = int(traded.sum())
    active_mask = pos_arr != 0

    result = {
        "backtest_df": df_aligned,
        "positions": positions,
        "horizon": horizon,
        "hold_periods": hold_periods,
        "n_trades": n_trades,
        "n_active_periods": int(active_mask.sum()),
        "summary": {},
    }

    for label, returns in [("gross", gross)] + [(f"net_{sc}", net_returns[sc]) for sc in cost_scenarios]:
        active_ret = returns[active_mask & ~np.isnan(returns)]
        cum = np.nancumsum(returns)

        if len(active_ret) > 1 and active_ret.std() > 0:
            sharpe = active_ret.mean() / active_ret.std() * np.sqrt(86400)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(cum)
        dd = np.where(peak != 0, (cum - peak) / np.abs(peak), 0)

        result["summary"][label] = {
            "total_pnl": float(np.nansum(returns)),
            "mean_return_per_trade": float(active_ret.mean()) if len(active_ret) > 0 else 0.0,
            "sharpe": float(sharpe),
            "hit_rate": float((active_ret > 0).sum() / len(active_ret)) if len(active_ret) > 0 else 0.0,
            "max_drawdown": float(dd.min()) if len(dd) > 0 else 0.0,
            "n_trades": n_trades,
        }

    logger.info(
        "Backtest %s: %d trades | gross Sharpe=%.2f | net(medium) Sharpe=%.2f",
        horizon, n_trades,
        result["summary"]["gross"]["sharpe"],
        result["summary"].get("net_medium", {}).get("sharpe", 0),
    )

    return result


def run_backtest_grid(
    df: pd.DataFrame,
    predictions_by_horizon: dict,
    horizons: list = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run backtests for multiple horizons and return a summary DataFrame.

    *predictions_by_horizon*: ``{"1s": pred_series, "5s": pred_series, ...}``
    """
    if horizons is None:
        horizons = list(predictions_by_horizon.keys())

    rows = []
    for h in horizons:
        if h not in predictions_by_horizon:
            continue
        result = run_backtest(df, predictions_by_horizon[h], horizon=h, config=config)
        for label, stats in result["summary"].items():
            row = {"horizon": h, "cost_scenario": label}
            row.update(stats)
            rows.append(row)

    return pd.DataFrame(rows)
