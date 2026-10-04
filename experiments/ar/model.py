"""## Executive summary (read this first)

Fit one AR(1) model per UST yield and simulate joint future yield paths.
This introductory model uses only daily level observations dated by the cutoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.base import BaselineForecaster, ForecastRequest, ForecastResult


class AR1Baseline(BaselineForecaster):
    """Each tenor predicts itself; correlated residuals form joint draws."""

    @property
    def model_name(self) -> str:
        return "ar1"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if request.target_type != "level":
            raise ValueError("This AR(1) example supports daily level targets only")
        if not request.horizons or min(request.horizons) < 1:
            raise ValueError("Daily horizons must be positive")

        panel = request.panels["rates_daily"].copy()
        asset_col = "asset" if "asset" in panel else "asset_id"
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel[panel["date"] <= pd.Timestamp(request.asof)]
        levels = (
            panel.pivot(index="date", columns=asset_col, values="value")
            .sort_index()
            .reindex(columns=request.asset_ids)
            .dropna()
        )
        if len(levels) < 30:
            raise ValueError("Need at least 30 complete pre-cutoff trading days")
        history = levels.to_numpy(dtype=float)
        if not np.isfinite(history).all():
            raise ValueError("Non-finite rate in training history")

        n_assets = len(request.asset_ids)
        intercept = np.empty(n_assets)
        slope = np.empty(n_assets)
        residuals = np.empty((len(history) - 1, n_assets))
        for i in range(n_assets):
            x = np.column_stack((np.ones(len(history) - 1), history[:-1, i]))
            y = history[1:, i]
            intercept[i], slope[i] = np.linalg.lstsq(x, y, rcond=None)[0]
            residuals[:, i] = y - (intercept[i] + slope[i] * history[:-1, i])

        covariance = np.atleast_2d(np.cov(residuals, rowvar=False))
        chol = np.linalg.cholesky(covariance + np.eye(n_assets) * 1e-8)
        rng = np.random.default_rng(1)
        state = np.tile(history[-1], (request.n_draws, 1))
        samples = np.empty(
            (request.n_draws, n_assets, len(request.horizons)), dtype=float
        )
        for day in range(1, max(request.horizons) + 1):
            shock = rng.standard_normal((request.n_draws, n_assets)) @ chol.T
            state = intercept + slope * state + shock
            for hi, horizon in enumerate(request.horizons):
                if day == horizon:
                    samples[:, :, hi] = state

        result = ForecastResult(
            samples=samples,
            asset_ids=request.asset_ids,
            horizons=request.horizons,
            model_name=self.model_name,
            metadata={
                "implementation": "ar1",
                "n_training_dates": len(levels),
                "training_cutoff": str(levels.index[-1].date()),
                "intercept": dict(zip(request.asset_ids, intercept.tolist())),
                "slope": dict(zip(request.asset_ids, slope.tolist())),
            },
        )
        self.validate_output(result, request)
        return result
