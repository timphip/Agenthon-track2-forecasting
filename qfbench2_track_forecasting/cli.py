"""Track-2 reference submission CLI.

Implements the `forecast` verb from the shared submission contract:

    forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD \
             --out /output/forecast.parquet

and writes the three deliverables the contract requires next to `--out`:

    forecast.parquet         the scored artifact — joint draws [draw, asset, horizon, value]
    forecast_meta.json       the sidecar g1_schema validates
    forecast_rationale.md    required, NEVER scored — the derivation, for human review

This is the statistical floor, not a worked example of using text. It reads the panels and
ignores `--text` entirely, which is stated plainly in the rationale it writes: a submission that
does this is doing the thing Track 2 exists to measure agents beating. It is here so that a
participant has something that provably builds, runs offline and passes g0-g3, and can be edited
into a real agent one step at a time.

Run offline. No network, no model weights, numpy + pandas only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd

from .horizons import HorizonMetadataError, monthly_horizon_steps
from .limits import ParseLimits
from .targets import log_return_steps

DEFAULT_DRAWS = 500
_RATIONALE_NAME = "forecast_rationale.md"


def _read_panels(panels_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Every parquet under --panels, keyed by filename stem.

    Accepts the contract layout (`/input/panels/*.parquet`) and also tolerates a unit that keeps
    its panels one level up, which is how the shipped exemplar was laid out before this CLI
    existed. Tolerating it here means a card authored either way still runs.
    """
    found = sorted(panels_dir.glob("*.parquet"))
    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no .parquet found under {panels_dir} (or its parent)")
    return {p.stem: pd.read_parquet(p) for p in found}


#: Both spellings occur in the shipped cards — the exemplar unit uses `asset_id`, the pilot and
#: prospective batches use `asset`. A reference implementation has to read either, or it works on
#: some cards and not others for a reason that has nothing to do with forecasting.
_ASSET_COLS = ("asset", "asset_id")


def _asset_col(df: pd.DataFrame) -> str | None:
    return next((c for c in _ASSET_COLS if c in df.columns), None)


def _diff_without_gaps(s: pd.Series) -> pd.Series:
    """First differences, with any difference that spans a hole in the data dropped.

    A transfer card ships its target asset as an early window plus a single row at the as-of
    date, with the years between deliberately withheld (the card says so, and says not to
    difference across it). Differenced naively, that hole reads as one day in which the asset
    moved a decade's worth -- on the CNY card it inflated the 5-95% band from under a percent
    to +-7%. The threshold adapts to the panel's own spacing (10x its typical step), so daily
    and monthly panels are both handled and a gapless panel is untouched.
    """
    d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    step = when.diff().dt.days
    if step.notna().sum() == 0:
        return d
    return d.where(step <= max(float(step.median()) * 10.0, 5.0))


def _series(panels: dict[str, pd.DataFrame], asset: str, asof: str) -> pd.Series:
    """The history of one asset up to and including the as-of, from whichever panel holds it."""
    for df in panels.values():
        col = _asset_col(df)
        if col is None:
            continue
        sub = df[df[col].astype(str) == asset]
        if sub.empty:
            continue
        sub = sub.copy()
        # Dates arrive as either strings or datetimes depending on how the panel was written.
        sub["date"] = sub["date"].astype(str).str.slice(0, 10)
        sub = sub[sub["date"] <= asof].sort_values("date")
        if not sub.empty:
            return sub.set_index("date")["value"].astype(float)
    seen = sorted(
        {
            str(v)
            for df in panels.values()
            if (c := _asset_col(df)) is not None
            for v in df[c].unique()
        }
    )
    raise SystemExit(
        f"asset {asset!r} not present in any panel at or before {asof}. "
        f"Panels carry: {', '.join(seen) if seen else '(no asset column found)'}"
    )


def _monthly_series(s: pd.Series) -> pd.Series:
    """Align monthly observations by period and refuse ambiguous source cadence."""
    out = s.copy()
    out.index = pd.to_datetime(out.index).to_period("M")
    if out.index.has_duplicates or len(out) < 3:
        raise HorizonMetadataError("Monthly target series need unique monthly observations.")
    if not np.isfinite(out.to_numpy()).all():
        raise HorizonMetadataError("Monthly target series contain non-finite observations.")
    if np.median(np.diff(out.index.asi8)) != 1:
        raise HorizonMetadataError("The selected target series do not have monthly cadence.")
    return out


def _daily_cadence(s: pd.Series) -> bool:
    """Recognize dense daily observations, not a duplicated or damaged monthly panel."""
    dates = pd.DatetimeIndex(pd.to_datetime(s.index))
    if len(dates) < 30 or dates.has_duplicates:
        return False
    gaps = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(float)
    per_month = pd.Series(1, index=dates.to_period("M")).groupby(level=0).sum()
    return bool(0 < np.median(gaps) <= 3 and per_month.median() >= 8)


def _explicit_monthly_periods(source: dict[str, Any]) -> bool:
    targets = source.get("targets", {})
    questions = source.get("questions", [])
    return (isinstance(targets, dict) and "observation_periods" in targets) or (
        isinstance(questions, list)
        and any(isinstance(row, dict) and "observation_period" in row for row in questions)
    )


def _monthly_inputs(
    panels: dict[str, pd.DataFrame],
    card: dict[str, Any],
    card_path: pathlib.Path,
    asof: str,
) -> np.ndarray | None:
    """Use explicit monthly task metadata; context-panel frequency never selects this path."""
    targets = card["targets"]
    frequency = targets.get("target_frequency", card.get("metadata", {}).get("target_frequency"))
    if frequency != "monthly":
        return None
    histories = {asset: _series(panels, asset, asof) for asset in targets["asset_ids"]}
    path = card_path.parent / "forecast_spec.json"
    spec = None
    if path.exists():
        try:
            spec = json.loads(path.read_text())
        except (ValueError, OSError):
            raise HorizonMetadataError(
                "Read a valid forecast_spec.json beside card.toml."
            ) from None
        if not isinstance(spec, dict):
            raise HorizonMetadataError("The forecast spec must be an object.")
    # Older month-ahead examples also call daily target observations "monthly".
    # Recover only when every selected series has dense daily observations and no
    # explicit monthly-period instructions; a damaged monthly series must still refuse.
    if all(_daily_cadence(history) for history in histories.values()):
        if _explicit_monthly_periods(card) or _explicit_monthly_periods(spec or {}):
            raise HorizonMetadataError(
                "Monthly observation-period metadata conflicts with daily target observations. "
                "Correct the task inputs."
            )
        warnings.warn(
            "The monthly frequency declaration conflicts with daily target observations; "
            "using daily sampling. Correct the task's target_frequency metadata.",
            UserWarning,
            stacklevel=2,
        )
        return None
    if targets.get("target_type", "level") != "level":
        raise HorizonMetadataError("The monthly reference sampler requires level targets.")
    last = {}
    for asset, history in histories.items():
        _monthly_series(history)
        last[asset] = str(history.index[-1])[:10]
    return monthly_horizon_steps(
        targets["asset_ids"],
        targets["horizons"],
        last,
        asof=asof,
        card=card,
        forecast_spec=spec,
    )


def _monthly_walk(
    rng: np.random.Generator,
    hist: dict[str, pd.Series],
    horizons: list[int],
    panel_steps: np.ndarray,
    last: np.ndarray,
    sd: np.ndarray,
    chol: np.ndarray,
    n_draws: int,
) -> np.ndarray:
    """Share each calendar month's correlated innovation across all requested horizons."""
    anchors = np.array([pd.Period(s.index[-1], freq="M").ordinal for s in hist.values()])
    endpoints = anchors[:, None] + panel_steps.astype(np.int64)
    path = np.zeros((n_draws, len(hist)))
    out = np.empty((n_draws, len(hist), len(horizons)))
    for month in range(int(anchors.min()) + 1, int(endpoints.max()) + 1):
        z = rng.standard_normal((n_draws, len(hist))) @ chol.T
        path += z * sd * (month > anchors)
        for ai, hi in np.argwhere(endpoints == month):
            out[:, ai, hi] = last[ai] + path[:, ai]
    return out


def _draw(
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    n_draws: int,
    seed: int,
    *,
    target_type: str = "level",
    panel_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint Gaussian walk, using level changes or daily log returns as steps.

    Drawing each asset independently would score badly on purpose: the composite puts 0.3 on the
    joint variogram term precisely to catch marginals that were stapled together. So the shared
    innovation is drawn from the empirical correlation of historical steps. Daily calls retain
    their existing sqrt(h) scaling. An explicit monthly step matrix selects cumulative paths
    in calendar months, including the panel publication lag.
    """
    rng = np.random.default_rng(seed)
    hist = {a: _series(panels, a, asof) for a in assets}
    returns_target = target_type == "log_return"
    monthly = panel_steps is not None
    if monthly:
        panel_steps = np.asarray(panel_steps, dtype=float)
        if (
            target_type != "level"
            or panel_steps.shape != (len(assets), len(horizons))
            or not np.isfinite(panel_steps).all()
            or np.any(panel_steps <= 0)
            or np.any(panel_steps != np.floor(panel_steps))
        ):
            raise HorizonMetadataError(
                "Provide one positive integer monthly step count per grid cell."
            )
        hist = {a: _monthly_series(s) for a, s in hist.items()}
    # Factor panels contain decimal simple returns. A cumulative log-return target sums
    # log(1+r) steps; differencing the rows or adding the last past return is incorrect.
    steps = pd.DataFrame(
        {a: pd.Series(log_return_steps(s), index=s.index) for a, s in hist.items()}
        if returns_target
        else {a: s.diff().where(np.r_[False, np.diff(s.index.asi8) == 1]) for a, s in hist.items()}
        if monthly
        else {a: _diff_without_gaps(s) for a, s in hist.items()}
    ).dropna()
    if len(steps) < 30:
        raise SystemExit(f"not enough history to estimate covariance ({len(steps)} rows)")

    last = (
        np.zeros(len(assets), dtype=float)
        if returns_target
        else np.array([hist[a].iloc[-1] for a in assets], dtype=float)
    )
    drift = steps.mean().to_numpy(dtype=float) if returns_target else np.zeros(len(assets))
    sd = steps.std().to_numpy(dtype=float)
    corr = steps.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # Nearest-PSD nudge: an empirical correlation can be indefinite after nan_to_num.
    w, v = np.linalg.eigh(corr)
    corr = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    chol = np.linalg.cholesky(corr)

    if monthly:
        panel_steps = cast(np.ndarray, panel_steps)
        out = _monthly_walk(rng, hist, horizons, panel_steps, last, sd, chol, n_draws)
        return out, {
            "last": {a: float(last[i]) for i, a in enumerate(assets)},
            "step_unit": "month",
            "step_sd": {a: float(sd[i]) for i, a in enumerate(assets)},
            "n_history_rows": int(len(steps)),
            "target_type": target_type,
            "panel_steps": {
                a: {str(h): int(panel_steps[i, j]) for j, h in enumerate(horizons)}
                for i, a in enumerate(assets)
            },
            "horizon_sd": {
                a: {
                    str(h): float(sd[i] * np.sqrt(panel_steps[i, j]))
                    for j, h in enumerate(horizons)
                }
                for i, a in enumerate(assets)
            },
        }
    out = np.empty((n_draws, len(assets), len(horizons)), dtype=float)
    for hi, h in enumerate(horizons):
        z = rng.standard_normal((n_draws, len(assets))) @ chol.T
        centre = drift * h if returns_target else last
        out[:, :, hi] = centre + z * (sd * np.sqrt(h))
    meta = {
        "last": {a: float(last[i]) for i, a in enumerate(assets)},
        "daily_sd": {a: float(sd[i]) for i, a in enumerate(assets)},
        "n_history_rows": int(len(steps)),
        "target_type": target_type,
        "daily_drift": {a: float(drift[i]) for i, a in enumerate(assets)},
    }
    return out, meta


def _rationale(
    unit_id: str,
    asof: str,
    assets: list[str],
    horizons: list[int],
    n_draws: int,
    stats: dict[str, Any],
    text_dir: pathlib.Path,
) -> str:
    n_docs = len(list(text_dir.glob("*.txt"))) if text_dir.is_dir() else 0
    if stats.get("step_unit") == "month":
        rows = "\n".join(
            f"| {a} | {h} | {stats['last'][a]:.4f} | {stats['panel_steps'][a][str(h)]} | "
            f"{stats['step_sd'][a]:.4f} | {stats['horizon_sd'][a][str(h)]:.4f} |"
            for a in assets
            for h in horizons
        )
        return f"""# Forecast rationale — {unit_id}

As of **{asof}**, monthly level forecasts at horizon keys {horizons}. {n_draws} joint draws.

## Anchor and scale

Each anchor is the last available monthly observation at or before the cutoff.
The panel can lag the as-of. Monthly steps include that publication lag and end at
its explicitly supplied observation period. The horizon key is unchanged.
The monthly standard deviation is estimated from consecutive monthly changes,
using {stats["n_history_rows"]} overlapping observations. No drift adjustment is made.

| asset | horizon key | anchor | monthly steps | monthly sd | sd at horizon |
|---|---|---|---|---|---|
{rows}

## Dependence and text

Correlated innovations are drawn once per calendar month and accumulated along
one path for each draw. Forecasts at later periods reuse the earlier innovations.
The marginal standard deviation is monthly sd times the square root of monthly steps.
No text adjustment is made. {n_docs} text document(s) were present and none was read.
"""
    returns_target = stats.get("target_type") == "log_return"
    anchor = (
        "Zero for every asset: the target sums log(1 + daily simple return) over the horizon. "
        "The last observed daily return belongs to the history, not to that future total."
        if returns_target
        else "The last observed value of each series at the as-of, taken from the shipped panels"
    )
    adjustments = (
        "The historical mean daily log return, multiplied by the horizon. This statistical drift "
        "uses only the supplied history at or before the as-of. No text adjustment is made."
        if returns_target
        else "**None.** This is a driftless random walk: the centre is the anchor, unadjusted. "
        "Every\n"
        "adjustment is zero and is listed as such rather than omitted, so the ledger below sums."
    )
    step_description = (
        "daily log returns, log(1 + panel value)" if returns_target else "first differences"
    )
    correlation_description = "daily log returns" if returns_target else "daily changes"
    ladder = "\n".join(
        f"| {a} | {stats['last'][a]:.4f} | {stats['daily_sd'][a]:.4f} | "
        f"{stats['daily_sd'][a] * np.sqrt(h):.4f} | {h} |"
        for a in assets
        for h in horizons
    )
    ledger_header = (
        "| asset | anchor | daily sd | sd at horizon | horizon (BD) |\n|---|---|---|---|---|"
    )
    centre_description = "Centre = anchor + 0 for every asset and horizon."
    if returns_target:
        ledger_header = (
            "| asset | anchor | daily drift | centre at horizon | daily sd | "
            "sd at horizon | horizon (BD) |\n"
            "|---|---|---|---|---|---|---|"
        )
        ladder = "\n".join(
            f"| {a} | 0.0000 | {stats['daily_drift'][a]:.4f} | "
            f"{stats['daily_drift'][a] * h:.4f} | {stats['daily_sd'][a]:.4f} | "
            f"{stats['daily_sd'][a] * np.sqrt(h):.4f} | {h} |"
            for a in assets
            for h in horizons
        )
        centre_description = "Centre = 0 + historical mean daily log return × horizon."
    return f"""# Forecast rationale — {unit_id}

As of **{asof}**, joint distribution over {", ".join(assets)} at horizon(s)
{", ".join(str(h) for h in horizons)} business days. {n_draws} draws.

## Anchor

{anchor}
({stats["n_history_rows"]} rows of overlapping daily history used for the covariance).

## Adjustments

{adjustments}

## Scale and shape

Per-asset daily standard deviation of {step_description}, scaled by sqrt(horizon). Gaussian
shape — deliberately not fat-tailed, since nothing here justifies a tail view.

The draws are **joint**: a single innovation vector is drawn per draw from the empirical
correlation of {correlation_description} across assets, so cross-asset structure is preserved
rather than independent marginals. The composite's variogram term scores that structure.

## Adjustment ledger

{ledger_header}
{ladder}

{centre_description}

## What the text corpus contributed

**Nothing.** {n_docs} document(s) were present at the text path and none was read. This is the
statistical floor a reasoning agent has to beat, not an example of using text — the whole point
of Track 2 is the gap between this and an agent that reads the corpus. A real submission would
use the documents to move the centre, skew the distribution, or widen the tails, and would say
here which document drove which adjustment and by how much.

## What would change this forecast

Any evidence at all. It currently uses none beyond the panel's own volatility.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="forecast",
        description="QFBench 2.0 Track-2 reference submission (statistical floor).",
    )
    p.add_argument("--panels", type=pathlib.Path, required=True)
    p.add_argument("--text", type=pathlib.Path, required=True)
    p.add_argument("--asof", required=True)
    p.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="path to forecast.parquet; the sidecars are written beside it",
    )
    p.add_argument(
        "--card",
        type=pathlib.Path,
        default=None,
        help="card.toml; defaults to <panels>/../card.toml. Supplies assets/horizons.",
    )
    p.add_argument("--n-draws", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    # --panels names the unit root (contract) but a card may still keep a panels/ subdir, so look
    # in the panels dir first and only then one level up. Deriving it as parent/ unconditionally
    # resolves to "/" when --panels is /input/, which is how this was wrong the first time.
    card_path = a.card
    if card_path is None:
        for cand in (a.panels / "card.toml", a.panels.parent / "card.toml"):
            if cand.exists():
                card_path = cand
                break
    if card_path is None or not card_path.exists():
        raise SystemExit(
            f"card.toml not found in {a.panels} or {a.panels.parent}; pass --card explicitly"
        )
    import tomllib

    card = tomllib.loads(card_path.read_text())
    tgt = card["targets"]
    assets = list(tgt["asset_ids"])
    horizons = [int(h) for h in tgt["horizons"]]
    unit_id = card["task"]["id"]
    # The card's `n_draws_min` is AUTHORITATIVE and was previously advisory: the reference
    # producer read it, the scorer never did, and the scorer instead compared the submission
    # against the participant's own declared `n_draws`. It is now a floor on both sides — this
    # producer honours it, and `limits.min_draws` enforces the contract floor in the scorer, in
    # code no missing module can skip.
    card_floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    floor = max(card_floor, DEFAULT_DRAWS, ParseLimits().min_draws)
    n_draws = max(a.n_draws or floor, floor)
    if n_draws > ParseLimits().max_draws:
        raise SystemExit(
            f"--n-draws {n_draws} exceeds the contract ceiling {ParseLimits().max_draws}; the "
            "scorer refuses a submission above it"
        )

    panels = _read_panels(a.panels)
    try:
        panel_steps = _monthly_inputs(panels, card, card_path, a.asof)
    except HorizonMetadataError as exc:
        raise SystemExit(str(exc)) from None
    # samples, stats = _draw(
    #     panels,
    #     assets,
    #     horizons,
    #     a.asof,
    #     n_draws,
    #     a.seed,
    #     target_type=tgt.get("target_type", "level"),
    #     panel_steps=panel_steps,
    # )

    use_ar = (
        "rates_daily" in panels
        and tgt.get("target_type", "level") == "level"
        and tgt.get(
            "target_frequency",
            card.get("metadata", {}).get("target_frequency"),
        ) == "daily"
    )

    if use_ar:
        from baselines.base import ForecastRequest
        from experiments.ar.model import AR1Baseline

        request = ForecastRequest(
            panels={"rates_daily": panels["rates_daily"]},
            asof=a.asof,
            asset_ids=assets,
            horizons=horizons,
            n_draws=n_draws,
        )
        result = AR1Baseline().forecast(request)
        samples, stats = result.samples, result.metadata
    else:
        samples, stats = _draw(
            panels,
            assets,
            horizons,
            a.asof,
            n_draws,
            a.seed,
            target_type=tgt.get("target_type", "level"),
            panel_steps=panel_steps,
        )

    out_dir = a.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"draw": d, "asset": asset, "horizon": h, "value": float(samples[d, ai, hi])}
            for d in range(n_draws)
            for ai, asset in enumerate(assets)
            for hi, h in enumerate(horizons)
        ]
    ).to_parquet(a.out, index=False)

    (out_dir / "forecast_meta.json").write_text(
        json.dumps(
            {
                "unit_id": unit_id,
                "asof": a.asof,
                "representation": "samples",
                "asset_ids": assets,
                "horizons": horizons,
                "n_draws": n_draws,
                "target": tgt.get("target_type", "level"),
                "rationale": {
                    "file": _RATIONALE_NAME,
                    # "method": "joint gaussian random walk, no text",
                    "method": (
                        "AR(1) with correlated residuals, no text"
                        if use_ar else "joint gaussian random walk, no text"
                    ),
                },
            },
            indent=2,
        )
        + "\n"
    )

    # (out_dir / _RATIONALE_NAME).write_text(
    #     _rationale(unit_id, a.asof, assets, horizons, n_draws, stats, a.text)
    # )
    if use_ar:
        rationale = (
            f"# Forecast rationale — {unit_id}\n\n"
            f"As of {a.asof}; {n_draws} joint forecast draws.\n\n"
            "## Method\n\n"
            "Fit one AR(1) to each daily rate series using observations "
            "no later than the as-of date. Simulate future values with "
            "correlated fitted residuals. No text was used.\n"
            )
    else:
        rationale = _rationale(
            unit_id, a.asof, assets, horizons, n_draws, stats, a.text
        )
    (out_dir / _RATIONALE_NAME).write_text(rationale)

    print(f"wrote {a.out.name}, forecast_meta.json and {_RATIONALE_NAME} to {out_dir}")
    print(f"  {len(assets)} asset(s) x {len(horizons)} horizon(s), {n_draws} draws")
    return 0


if __name__ == "__main__":
    sys.exit(main())
