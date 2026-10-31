"""
Per-protein turnover fitting (vectorised).

For each protein a single turnover parameter is fitted by least squares against
the observed Label/Total ratios (Pulse only, or Pulse and Pulse&Chase jointly),
then converted to a half-life:

    single-exponential :  parameter k    (rate),          half-life = ln(2) / k
    2-compartment      :  parameter tau  (time constant), half-life = ln(2) * tau

The residual is ``sum((pred - obs)**2)`` over the finite observed points, exactly
as in the MATLAB objective functions F300_def_Obj2 / F400_def_Obj2 (the prepended
t = 0 point, where both prediction and observation are 0, does not affect the fit).

Because the prediction depends only on the fitted parameter (and the fixed time
vector / precursor), it is evaluated once on a dense log-spaced parameter grid and
every protein is then solved by matrix arithmetic: grid minimisation + a 3-point
parabolic refine.  Standard errors and 95 % confidence intervals come from the
local curvature of the sum-of-squares (asymptotic linearisation + delta method).
This fits thousands of proteins in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import models

LN2 = np.log(2.0)
_LN10 = np.log(10.0)

# Bounds on the fitted parameter, expressed as a half-life window (days).
_HALFLIFE_MIN = 0.05
_HALFLIFE_MAX = 5000.0
_N_GRID = 400

# t-distribution 97.5 % quantiles for small dof (two-sided 95 % CI).
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}


def _t_quantile(dof):
    dof = np.asarray(dof, dtype=float)
    keys = np.array(sorted(_T_TABLE))
    vals = np.array([_T_TABLE[k] for k in keys])
    out = np.interp(dof, keys, vals, left=_T_TABLE[keys.min()], right=1.96)
    out = np.where(dof > 60, 1.96, out)
    out = np.where(dof <= 0, np.nan, out)
    return out


@dataclass
class FitResult:
    index: int
    protein_group: str
    gene: str
    description: str
    half_life: float
    half_life_se: float
    ci_low: float
    ci_high: float
    parameter: float            # k or tau in the model's native units
    r2: float
    sse: float
    n_points: int
    model: str
    fit_mode: str               # 'pulse' or 'pulse+chase'
    note: str = ""

    def as_row(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
#  prediction grid
# --------------------------------------------------------------------------- #
def build_prediction_grid(t, tpoint, model, precursor, fit_pandc, n_grid=_N_GRID):
    """
    Evaluate the model on a dense log-spaced parameter grid (done once per run).

    Returns ``(grid, U, pred_pulse, pred_chase)`` where ``U = log10(grid)`` and
    the prediction arrays have shape ``(n_grid, len(t))``.  ``pred_chase`` is
    ``None`` when ``fit_pandc`` is False.
    """
    if model in ("single", "single_exponential"):
        p_lo, p_hi = LN2 / _HALFLIFE_MAX, LN2 / _HALFLIFE_MIN     # rate k
    else:
        p_lo, p_hi = _HALFLIFE_MIN / LN2, _HALFLIFE_MAX / LN2     # tau

    grid = np.logspace(np.log10(p_lo), np.log10(p_hi), n_grid)
    U = np.log10(grid)
    pred_pulse = models.predict_grid(grid, t, tpoint, chase=False, model=model, precursor=precursor)
    pred_chase = None
    if fit_pandc:
        pred_chase = models.predict_grid(grid, t, tpoint, chase=True, model=model, precursor=precursor)
    # replace non-finite predictions with a large sentinel so they lose the fit
    pred_pulse = np.where(np.isfinite(pred_pulse), pred_pulse, 1e6)
    if pred_chase is not None:
        pred_chase = np.where(np.isfinite(pred_chase), pred_chase, 1e6)
    return grid, U, pred_pulse, pred_chase


def total_sse_for_precursor(t, pulse_ratios, pandc_ratios, tpoint, model,
                            precursor, fit_pandc, n_grid=160):
    """
    Sum over all proteins of the minimum per-protein SSE for a given precursor.

    This is the objective minimised by the paper-style global precursor search:
    each protein is assigned its best-fit turnover on a coarse grid and the
    residuals are summed. Lean (no CI / R^2) and fully vectorised.
    """
    op = np.asarray(pulse_ratios, dtype=float)
    N, T = op.shape
    grid, U, PP, PC = build_prediction_grid(t, tpoint, model, precursor, fit_pandc, n_grid)
    G = grid.size

    mp = np.isfinite(op)
    opf = np.where(mp, op, 0.0)
    SSE = np.zeros((N, G))
    for k in range(T):
        diff = PP[:, k][None, :] - opf[:, k][:, None]
        SSE += mp[:, k][:, None] * diff * diff
    if fit_pandc and pandc_ratios is not None:
        oc = np.asarray(pandc_ratios, dtype=float)
        mc = np.isfinite(oc)
        ocf = np.where(mc, oc, 0.0)
        for k in range(T):
            diff = PC[:, k][None, :] - ocf[:, k][:, None]
            SSE += mc[:, k][:, None] * diff * diff
    per_protein_min = SSE.min(axis=1)
    return float(np.sum(per_protein_min))


# --------------------------------------------------------------------------- #
#  batch fit
# --------------------------------------------------------------------------- #
def fit_batch(t, pulse_ratios, pandc_ratios, tpoint, model, precursor, fit_pandc,
              labels=None, n_grid=_N_GRID, progress=None):
    """
    Fit every protein at once.

    Parameters
    ----------
    t             : time vector (T,), includes the prepended 0.
    pulse_ratios  : (N, T) observed Pulse ratios.
    pandc_ratios  : (N, T) observed Pulse&Chase ratios, or None.
    labels        : list of (protein_group, gene, description) tuples.

    Returns list[FitResult].
    """
    t = np.asarray(t, dtype=float)
    op = np.asarray(pulse_ratios, dtype=float)
    N, T = op.shape
    G = n_grid

    grid, U, PP, PC = build_prediction_grid(t, tpoint, model, precursor, fit_pandc, n_grid)
    h = U[1] - U[0]

    mp = np.isfinite(op)
    opf = np.where(mp, op, 0.0)

    if fit_pandc and pandc_ratios is not None:
        oc = np.asarray(pandc_ratios, dtype=float)
        mc = np.isfinite(oc)
        ocf = np.where(mc, oc, 0.0)
    else:
        oc = mc = ocf = None

    # ---- SSE(N, G) accumulated column-by-time (memory friendly) ----
    SSE = np.zeros((N, G))
    for k in range(T):
        diff = PP[:, k][None, :] - opf[:, k][:, None]      # (N, G)
        SSE += mp[:, k][:, None] * diff * diff
    if oc is not None:
        for k in range(T):
            diff = PC[:, k][None, :] - ocf[:, k][:, None]
            SSE += mc[:, k][:, None] * diff * diff
        if progress:
            progress(N, N)

    rows = np.arange(N)
    gstar = np.argmin(SSE, axis=1)
    gm = np.clip(gstar, 1, G - 2)
    Sm1 = SSE[rows, gm - 1]
    S0 = SSE[rows, gm]
    Sp1 = SSE[rows, gm + 1]
    curv = Sm1 - 2.0 * S0 + Sp1

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (Sm1 - Sp1) / curv          # vertex offset in units of h
    delta = np.where(curv > 0, np.clip(delta, -1.0, 1.0), 0.0)
    u_star = U[gm] + delta * h
    param = np.power(10.0, u_star)

    # effective points (exclude the prepended t = 0, matching MATLAB's zero filter)
    n_eff = mp.sum(axis=1) - mp[:, 0].astype(int)
    if oc is not None:
        n_eff = n_eff + mc.sum(axis=1) - mc[:, 0].astype(int)
    dof = np.maximum(n_eff - 1, 1)

    b = curv / (2.0 * h * h)                        # SSE ~= a + b (u - u*)^2
    sse_min = np.maximum(S0 - b * (delta * h) ** 2, 0.0)
    sigma2 = sse_min / dof

    # Sum of squared prediction gradients wrt the native parameter,
    # derived from the parabola: sum g^2 = b / (param * ln10)^2.
    with np.errstate(divide="ignore", invalid="ignore"):
        sumg2 = b / (param * _LN10) ** 2
        var_param = np.where(sumg2 > 0, sigma2 / sumg2, np.nan)
    se_param = np.sqrt(var_param)

    half_life = np.asarray(models.param_to_halflife(param, model), dtype=float)
    if model in ("single", "single_exponential"):
        with np.errstate(divide="ignore", invalid="ignore"):
            dhl = LN2 / (param ** 2)
    else:
        dhl = np.full_like(param, LN2)
    hl_se = np.abs(dhl) * se_param

    tval = _t_quantile(dof)
    ci_low = half_life - tval * hl_se
    ci_high = half_life + tval * hl_se

    # ---- R^2 using the interpolated prediction at u_star ----
    neigh = np.clip(gm + np.sign(delta).astype(int), 0, G - 1)
    w = np.abs(delta)[:, None]
    pred_p = (1 - w) * PP[gm] + w * PP[neigh]        # (N, T)
    obs_list = [opf * mp]
    pred_list = [pred_p * mp]
    mask_list = [mp]
    if oc is not None:
        pred_c = (1 - w) * PC[gm] + w * PC[neigh]
        obs_list.append(ocf * mc)
        pred_list.append(pred_c * mc)
        mask_list.append(mc)
    r2 = _r2_rows(obs_list, pred_list, mask_list)

    # ---- bounds / quality notes ----
    p_lo, p_hi = grid[0], grid[-1]
    at_lower_param = param <= p_lo * 1.0001         # long half-life (single) / short (2comp)
    at_upper_param = param >= p_hi * 0.9999

    labels = labels or [("", "", "")] * N
    fit_mode = "pulse+chase" if fit_pandc else "pulse"
    results = []
    for i in range(N):
        pg, gene, desc = labels[i] if i < len(labels) else ("", "", "")
        note = ""
        if n_eff[i] < 1:
            note = "insufficient data"
        elif model in ("single", "single_exponential"):
            if at_lower_param[i]:
                note = "half-life at upper bound (very slow / no turnover)"
            elif at_upper_param[i]:
                note = "half-life at lower bound (very fast turnover)"
        else:
            if at_lower_param[i]:
                note = "half-life at lower bound (very fast turnover)"
            elif at_upper_param[i]:
                note = "half-life at upper bound (very slow / no turnover)"
        if not np.isfinite(hl_se[i]) and not note:
            note = "SE not estimable"

        results.append(FitResult(
            index=i, protein_group=pg, gene=gene, description=desc,
            half_life=float(half_life[i]) if n_eff[i] >= 1 else float("nan"),
            half_life_se=float(hl_se[i]),
            ci_low=float(ci_low[i]), ci_high=float(ci_high[i]),
            parameter=float(param[i]) if n_eff[i] >= 1 else float("nan"),
            r2=float(r2[i]), sse=float(sse_min[i]),
            n_points=int(n_eff[i]), model=model, fit_mode=fit_mode, note=note))
    return results


def fit_protein(index, t, obs_pulse, obs_pandc, tpoint, model="single",
                precursor=None, fit_pandc=False, labels=("", "", "")):
    """Convenience wrapper: fit a single protein (used for detail views)."""
    op = np.asarray(obs_pulse, dtype=float)[None, :]
    oc = None if (obs_pandc is None) else np.asarray(obs_pandc, dtype=float)[None, :]
    res = fit_batch(t, op, oc, tpoint, model, precursor, fit_pandc,
                    labels=[labels])[0]
    res.index = index
    return res


def _r2_rows(obs_list, pred_list, mask_list):
    """Row-wise R^2 across the concatenated (masked) observation blocks."""
    obs = np.concatenate(obs_list, axis=1)
    pred = np.concatenate(pred_list, axis=1)
    mask = np.concatenate(mask_list, axis=1).astype(float)
    n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (obs * mask).sum(axis=1) / np.maximum(n, 1)
        ss_res = ((obs - pred) ** 2 * mask).sum(axis=1)
        ss_tot = ((obs - mean[:, None]) ** 2 * mask).sum(axis=1)
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where((ss_tot > 0) & (n >= 2), r2, np.nan)
    return r2
