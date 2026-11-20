"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Model: per-trophy saturating growth in logit space.
  logit(rate(t)) = A - B * exp(-k * t),  t = hours since 2026-11-19T00:00:00Z
fitted by least squares over a log-spaced grid of k, with the asymptote A
clamped to a per-category prior interval (see PRIORS, derived from the
trophy guide's kind/stage/grade per trophy id) so a 2-day history cannot
run away when extrapolated 4.5 days to the target. B >= 0 (rates only rise
along the fitted curve); the forecast median is floored at last_rate - 0.5
logit so a noisy fit never predicts a big drop below the last read.

Spread: honest logit sigma from fit residual + horizon growth + category
uncertainty (missable / platinum / gold / collectibles get extra width);
low-count, jumpy trophies get wide bands, settled ones narrow bands.
A fit on fewer than 6 points gets extra widening (+0.15 logit per
missing point): a 3-parameter curve is nearly exact on 3-5 points by
construction, so its residual understates the real uncertainty.

Flat-history branch: when every read in the case is identical (the page
had not refreshed yet, e.g. the practice cases), the observed rates are
stale pre-refresh values. Exophase refreshes dilute rates as the tracked
player base jumps, so the median is shifted down 1.0 logit and the band
widened (sigma 1.0) instead of trusting the stale plateau. Why 1.0: the
documented launch-night refresh grew the base 638 -> 3868 players (~6x),
and the house's own persistence practice loss (1.51) sits near the
frozen-earner end of that dilution, i.e. new players had earned almost
nothing yet; -0.5 would assume they earn at half the old rate. A
simulation over refresh depths (new players earning 0%/10%/25%/50% of
the old rate) puts -1.0/sigma 1.0 ahead of -0.5/sigma 0.85 under every
depth (worst case 0.42 vs 0.72 pinball loss).

Refresh-regime split: a read where players_tracked jumps >25% marks an
Exophase refresh — the tracked base jumps and every rate dilutes at once.
The saturating fit runs on the segment after the last refresh when that
segment has 3+ distinct times; otherwise it falls back to the whole
history. Pre-refresh reads describe a smaller population, and fitting
across the dilution drop would bias the curve and inflate residuals.

Deterministic: pure arithmetic, no randomness, no clock, no network.
Reads every <id>.in in the input dir (name order), writes <id>.out.

Run locally:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Category priors -------------------------------------------------------
# Per-trophy asymptote interval (percent) for the saturating curve, from the
# trophy guide (kind / stage / grade per id). Story trophies saturate high
# and steep; collectibles / grind slow and low; missable diverges mid-way;
# platinum and the 100%-completion gold saturate very low by day 7.
# Key: trophy id -> (group, asymptote_lo_pct, asymptote_hi_pct).
PRIORS = {
    1: ("platinum", 0.1, 5.0),
    2: ("story", 40.0, 99.0), 3: ("story", 40.0, 99.0),
    4: ("story", 40.0, 99.0), 5: ("story", 40.0, 99.0),
    6: ("story", 40.0, 99.0), 7: ("story-final", 2.0, 40.0),
    8: ("story", 40.0, 99.0), 9: ("story", 40.0, 99.0),
    10: ("story", 40.0, 99.0), 11: ("story", 40.0, 99.0),
    12: ("story", 40.0, 99.0), 13: ("story", 40.0, 99.0),
    14: ("completion", 0.5, 15.0),
    15: ("collectibles", 1.0, 40.0), 16: ("activities", 5.0, 70.0),
    17: ("activities", 5.0, 70.0), 18: ("activities", 5.0, 70.0),
    19: ("activities", 5.0, 70.0), 20: ("activities", 5.0, 70.0),
    21: ("activities", 5.0, 70.0), 22: ("side", 10.0, 70.0),
    23: ("side", 10.0, 70.0), 24: ("side", 10.0, 70.0),
    25: ("side", 10.0, 70.0), 26: ("side", 10.0, 70.0),
    27: ("side", 10.0, 70.0), 28: ("missable", 2.0, 50.0),
    29: ("side", 10.0, 70.0), 30: ("collectibles", 1.0, 40.0),
    31: ("collectibles", 1.0, 40.0), 32: ("activities", 5.0, 70.0),
    33: ("misc", 2.0, 70.0), 34: ("robbery", 5.0, 60.0),
    35: ("collectibles", 1.0, 40.0), 36: ("misc", 2.0, 70.0),
    37: ("misc", 2.0, 70.0), 38: ("misc", 2.0, 70.0),
    39: ("robbery", 5.0, 60.0), 40: ("robbery", 5.0, 60.0),
    41: ("robbery", 5.0, 60.0), 42: ("collectibles", 1.0, 40.0),
    43: ("robbery", 5.0, 60.0), 44: ("misc", 2.0, 70.0),
    45: ("robbery", 5.0, 60.0), 46: ("robbery", 5.0, 60.0),
    47: ("activities", 5.0, 70.0), 48: ("misc", 2.0, 70.0),
    49: ("misc", 2.0, 70.0), 50: ("misc", 2.0, 70.0),
    51: ("misc", 2.0, 70.0), 52: ("robbery", 5.0, 60.0),
    53: ("misc", 2.0, 70.0), 54: ("collectibles", 1.0, 40.0),
    55: ("misc", 2.0, 70.0),
}

# Extra logit spread per group (honest uncertainty on top of fit + horizon).
GROUP_SPREAD = {
    "platinum": 0.15, "story-final": 0.15, "completion": 0.15,
    "missable": 0.25, "collectibles": 0.15,
}

DEFAULT_PRIOR = ("misc", 2.0, 70.0)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return (dt - RELEASE).total_seconds() / 3600.0


def _wls_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (sum(ys) / len(ys) if ys else 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def fit_saturation(ts, ys, ws, a_lo, a_hi):
    """Fit y = A - B*exp(-k*t) with A in [a_lo, a_hi] (logit), B >= 0.

    Grid-searches k; for each k fits A, B by weighted least squares, clamps
    A into its prior interval and refits B with A fixed. Returns
    (A, B, k, resid_std) of the best (lowest SSE) constrained fit.
    """
    best = None
    # 48 log-spaced k values, 0.002 .. 1.5 per hour.
    n_grid = 48
    lk0, lk1 = math.log(0.002), math.log(1.5)
    for gi in range(n_grid):
        k = math.exp(lk0 + (lk1 - lk0) * gi / (n_grid - 1))
        xs = [math.exp(-k * t) for t in ts]
        # y = A - B*x  -> linear fit y = a + b*x with b = -B.
        a, b = _wls_line(xs, ys, ws)
        if b > 0:  # B < 0: refit with B = 0 (flat at mean).
            a, b = sum(w * y for w, y in zip(ws, ys)) / sum(ws), 0.0
        if a < a_lo:
            a = a_lo
            # Refit B with A fixed: minimise sum w*(y - (A - B*x))^2.
            num = sum(w * x * (a - y) for w, x, y in zip(ws, xs, ys))
            den = sum(w * x * x for w, x in zip(ws, xs))
            b = -num / den if den > 1e-12 else 0.0
            if b > 0:
                b = 0.0
        elif a > a_hi:
            a = a_hi
            num = sum(w * x * (a - y) for w, x, y in zip(ws, xs, ys))
            den = sum(w * x * x for w, x in zip(ws, xs))
            b = -num / den if den > 1e-12 else 0.0
            if b > 0:
                b = 0.0
        sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        if best is None or sse < best[0]:
            best = (sse, a, -b, k)
    sse, a, B, k = best
    dof = max(1, len(ts) - 3)
    resid = math.sqrt(sse / sum(ws) / dof) if sum(ws) > 0 else 0.0
    return a, B, k, resid


def _last_refresh_index(history):
    """Index of the first read after the last Exophase refresh, or 0.

    A refresh is a read where players_tracked jumps >25% over the
    previous read: Exophase re-pulls the page, the tracked base jumps,
    and every rate dilutes at once. Reads before a refresh describe a
    different, smaller population.
    """
    idx = 0
    prev = None
    for i, read in enumerate(history):
        players = read.get("players_tracked") or 0
        if prev and players > prev * 1.25:
            idx = i
        prev = players
    return idx


def _collect_points(history, idx):
    """Collect (t, logit rate, weight) points, collapsing consecutive
    duplicate rates (a stale page repeats the same value; keep the last
    time of each run)."""
    pts = []
    last_y = None
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        y = logit(rate)
        players = read.get("players_tracked") or 1
        w = math.sqrt(max(1.0, float(players)))
        t = hours_since_release(parse_time(read["at"]))
        if last_y is not None and abs(y - last_y) < 1e-12:
            # Same value as the previous point: extend that run's time.
            pts[-1] = (t, y, pts[-1][2])
            continue
        pts.append((t, y, w))
        last_y = y
    return pts


def forecast_trophy(idx, trophy, history, horizon_minutes, target_dt):
    """Return the 5 quantiles (percent) for one trophy."""
    group, lo_pct, hi_pct = PRIORS.get(trophy["id"], DEFAULT_PRIOR)
    a_lo, a_hi = logit(lo_pct), logit(hi_pct)
    extra_spread = GROUP_SPREAD.get(group, 0.0)
    horizon_days = horizon_minutes / 1440.0
    t_target = hours_since_release(target_dt)

    # Refresh regime: fit on the segment after the last refresh when it
    # has enough movement to fit; pre-refresh reads describe a smaller
    # population, and the dilution drop at the refresh would otherwise
    # bias the saturating fit and inflate its residual.
    seg_start = _last_refresh_index(history)
    pts = _collect_points(history[seg_start:], idx)
    if len({round(p[0], 6) for p in pts}) < 3 and seg_start > 0:
        pts = _collect_points(history, idx)

    if not pts:
        # No data at all: flat 0.01% with a wide band.
        med = logit(0.01)
        return quantiles(med, 1.5)

    last_rate = inv_logit(pts[-1][1])
    last_logit = pts[-1][1]
    distinct_t = len({round(p[0], 6) for p in pts})
    all_flat = len(pts) == 1

    if all_flat or distinct_t < 3:
        # No movement to fit: stale pre-refresh plateau. Persistence median
        # shifted down for refresh dilution, with a wide band.
        med = last_logit - 1.0
        return quantiles(med, 1.0)

    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ws = [p[2] for p in pts]
    A, B, _k, resid = fit_saturation(ts, ys, ws, a_lo, a_hi)
    med = A - B * math.exp(-_k * t_target)
    # Earn rates do not fall along the curve; never predict far below the
    # last read (allow a small dilution dip only).
    med = max(med, last_logit - 0.5)
    med = min(max(med, logit(0.01)), logit(99.99))

    # Spread: fit residual (grows with extrapolation) + horizon random
    # walk + category uncertainty. Wide when the fit is jumpy or the
    # trophy is low-count; narrow when settled.
    resid_c = min(1.0, max(0.08, resid))
    s = math.sqrt(
        (resid_c * (1.0 + 0.15 * horizon_days)) ** 2
        + (0.22 * math.sqrt(horizon_days)) ** 2
        + extra_spread ** 2
    )
    # Few points -> the 3-parameter saturating fit is nearly exact by
    # construction (3 free parameters on 3-5 points), so its residual
    # understates the real uncertainty. Widen the band for short
    # post-refresh segments; a long history earns the narrow band.
    if len(pts) < 6:
        s += 0.15 * (6 - len(pts))
    s = min(2.0, max(0.20, s))
    return quantiles(med, s)


def quantiles(med_logit, sigma):
    qs = []
    for z in Z:
        q = clamp_pct(inv_logit(med_logit + z * sigma))
        qs.append(round4(q))
    # Enforce non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    target_dt = parse_time(case["target"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        q = forecast_trophy(i, trophy, case["history"],
                            case["horizon_minutes"], target_dt)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
