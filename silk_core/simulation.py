"""
Noise-driven simulation to justify the reliable half-life range.

The experiment samples the protein labelling curve at only a few early time
points (e.g. 0, 5, 14, 21 days), yet half-lives of tens to > 100 days are
reported.  This module answers, *with the study's own measurement noise*, the
question a reviewer will ask: **up to what half-life can this sampling design
actually recover a half-life without bias and with acceptable precision?**

Method (a parametric / Monte-Carlo recovery study):

1. Take the precursor labelling curve and kinetic model *exactly as fitted from
   the real data* (:class:`silk_core.pipeline.RunResult`).
2. For each *true* half-life in a user-chosen set (default 20, 50, 75, 100, 150,
   200 days) compute the noiseless labelling curve at the real sampling times.
3. Add the **measured** noise (per-time-point residual SD estimated from the
   real good-fitting proteins) and the **measured** missing-data pattern.
4. Re-fit the half-life from those noisy samples with the *same* fitting code
   used on the real data, and record the estimate and its 95 % CI.
5. Over many replicates, report for each true half-life: bias, relative error,
   95 % CI coverage and CI width.

:func:`recommend_cutoff` then turns the coverage/bias/error curves into a
data-driven statement: the largest half-life recovered within tolerance (a hard
cutoff), or -- if precision degrades gradually -- a "~X days" soft bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import models, fitting


DEFAULT_TRUE_HALFLIVES = (0.5, 1.0, 5.0, 20.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0)


# --------------------------------------------------------------------------- #
#  noise / missingness estimated from the real data
# --------------------------------------------------------------------------- #
@dataclass
class NoiseModel:
    times: np.ndarray            # measured (non-zero) design time points
    sd_pulse: np.ndarray         # per-time-point residual SD (pulse)
    sd_chase: np.ndarray | None  # per-time-point residual SD (chase) or None
    missing_pulse: np.ndarray    # per-time-point missing fraction (pulse)
    missing_chase: np.ndarray | None
    pooled_sd: float
    n_used: int
    robust: bool
    selection: str = "analysis_range"
    hl_range: tuple = (1.0, 100.0)
    n_total: int = 0

    def selection_label(self) -> str:
        if self.selection == "analysis_range":
            return (f"proteins within the analysis range "
                    f"({self.hl_range[0]:g}-{self.hl_range[1]:g} d)")
        if self.selection == "all":
            return "all fitted proteins"
        return "well-fitting proteins (R2 filter, not at a fit bound)"

    def describe(self) -> str:
        pairs = ", ".join(f"{tv:g}d:{sd:.4f}" for tv, sd in zip(self.times, self.sd_pulse))
        return (f"per-time-point noise SD (pulse) [{pairs}]; "
                f"pooled SD={self.pooled_sd:.4f}; estimated from {self.n_used} of "
                f"{self.n_total} {self.selection_label()} "
                f"({'robust MAD' if self.robust else 'std'}).")


NOISE_SELECTIONS = {
    "analysis_range": "Analysis range (half-life within the range, no R2 filter)",
    "all": "All fitted proteins",
    "r2": "R2-filtered (R2 >= r2_min, not at a fit bound)",
}


def estimate_noise_model(result, selection="analysis_range", hl_range=(1.0, 100.0),
                         r2_min=0.8, robust=True) -> NoiseModel:
    """Estimate per-time-point measurement noise and missingness from the run.

    Residuals (observed - fitted) are collected from a chosen protein set and
    their per-time-point spread is the measurement-noise estimate:

    * ``'analysis_range'`` (default) -- every protein whose fitted half-life lies
      within ``hl_range`` (inclusive), i.e. the proteins actually carried into the
      downstream analysis.  No goodness-of-fit (R2) filter is applied.
    * ``'all'``  -- every protein with a finite fit.
    * ``'r2'``   -- legacy: ``R2 >= r2_min`` and not pinned to a fit bound.

    Missing fractions come from the raw data.
    """
    if selection not in NOISE_SELECTIONS:
        raise ValueError(f"selection must be one of {list(NOISE_SELECTIONS)}")
    t = np.asarray(result.protein_pulse.time, dtype=float)
    meas = t > 0
    meas_times = t[meas]
    tpoint = result.tpoint
    model = result.model
    prec = result.precursor_params

    pulse = np.asarray(result.protein_pulse.ratios, dtype=float)
    fit_pandc = result.fit_mode == "pulse+chase" and getattr(result, "pandc_aligned", None) is not None
    pandc = np.asarray(result.pandc_aligned, dtype=float) if fit_pandc else None

    fr = result.fit_results
    params = np.array([f.parameter for f in fr], dtype=float)
    r2 = np.array([f.r2 for f in fr], dtype=float)
    atbound = np.array([f.note.startswith("half-life at") for f in fr])
    hl = np.array([f.half_life for f in fr], dtype=float)
    finite = np.isfinite(params) & np.isfinite(hl)
    if selection == "analysis_range":
        lo, hi = float(hl_range[0]), float(hl_range[1])
        good = finite & (hl >= lo) & (hl <= hi)
    elif selection == "all":
        good = finite
    else:
        good = finite & np.isfinite(r2) & (r2 >= r2_min) & (~atbound)
    gidx = np.where(good)[0]

    def _spread(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        if robust:
            return float(1.4826 * np.median(np.abs(vals - np.median(vals))))
        return float(np.std(vals, ddof=1))

    resid_cols_p = {c: [] for c in np.where(meas)[0]}
    resid_cols_c = {c: [] for c in np.where(meas)[0]} if fit_pandc else None
    for i in gidx:
        pr = models.predict(params[i], t, tpoint, chase=False, model=model, precursor=prec)
        obs = pulse[i]
        for c in np.where(meas)[0]:
            if np.isfinite(obs[c]) and np.isfinite(pr[c]):
                resid_cols_p[c].append(obs[c] - pr[c])
        if fit_pandc:
            prc = models.predict(params[i], t, tpoint, chase=True, model=model, precursor=prec)
            oc = pandc[i]
            for c in np.where(meas)[0]:
                if np.isfinite(oc[c]) and np.isfinite(prc[c]):
                    resid_cols_c[c].append(oc[c] - prc[c])

    sd_pulse = np.array([_spread(resid_cols_p[c]) for c in np.where(meas)[0]])
    sd_chase = (np.array([_spread(resid_cols_c[c]) for c in np.where(meas)[0]])
                if fit_pandc else None)

    all_res = [v for c in resid_cols_p for v in resid_cols_p[c]]
    pooled = _spread(all_res)
    # backfill any non-finite per-point SD with the pooled value
    sd_pulse = np.where(np.isfinite(sd_pulse), sd_pulse, pooled)
    if sd_chase is not None:
        sd_chase = np.where(np.isfinite(sd_chase), sd_chase, pooled)

    miss_pulse = np.array([1.0 - np.isfinite(pulse[:, c]).mean() for c in np.where(meas)[0]])
    miss_chase = (np.array([1.0 - np.isfinite(pandc[:, c]).mean() for c in np.where(meas)[0]])
                  if fit_pandc else None)

    return NoiseModel(times=meas_times, sd_pulse=sd_pulse, sd_chase=sd_chase,
                      missing_pulse=miss_pulse, missing_chase=miss_chase,
                      pooled_sd=float(pooled), n_used=int(gidx.size), robust=robust,
                      selection=selection, hl_range=(float(hl_range[0]), float(hl_range[1])),
                      n_total=len(fr))


# --------------------------------------------------------------------------- #
#  the recovery simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    table: pd.DataFrame                     # one row per true half-life
    raw: dict = field(default_factory=dict) # true_hl -> dict(est, ci_low, ci_high)
    noise: NoiseModel | None = None
    model: str = "2comp"
    fit_mode: str = "pulse"
    times: tuple = ()
    n_rep: int = 0
    noise_scale: float = 1.0
    extra_missing: float = 0.0
    seed: int = 0


def simulate_recovery(result, true_halflives=DEFAULT_TRUE_HALFLIVES, n_rep=2000,
                      noise=None, noise_scale=1.0, extra_missing=0.0,
                      seed=0, use_chase=None, r2_min=0.8, robust=True,
                      noise_selection="analysis_range", hl_range=(1.0, 100.0),
                      progress=None) -> SimulationResult:
    """Monte-Carlo recovery of known half-lives under the study's noise.

    Parameters
    ----------
    result          : a completed RunResult (supplies the precursor curve, model,
                      fit mode, sampling times and tpoint).
    true_halflives  : iterable of ground-truth half-lives (days).
    n_rep           : simulated proteins per true half-life.
    noise           : a NoiseModel; estimated from ``result`` when None.
    noise_scale     : multiply the estimated noise SD (sensitivity analysis).
    extra_missing   : additional missing probability per point, on top of the
                      measured missingness (sensitivity analysis).
    seed            : RNG seed (reproducible).
    use_chase       : force pulse+chase (True) / pulse-only (False); default
                      follows ``result.fit_mode``.

    Returns
    -------
    SimulationResult
    """
    t = np.asarray(result.protein_pulse.time, dtype=float)
    meas = t > 0
    meas_idx = np.where(meas)[0]
    tpoint = result.tpoint
    model = result.model
    prec = result.precursor_params

    fit_pandc = (result.fit_mode == "pulse+chase") if use_chase is None else bool(use_chase)
    if fit_pandc and getattr(result, "pandc_aligned", None) is None:
        fit_pandc = False

    if noise is None:
        noise = estimate_noise_model(result, selection=noise_selection, hl_range=hl_range,
                                     r2_min=r2_min, robust=robust)

    # map the noise SD onto the design columns (align by time)
    sd_p = np.zeros(t.size)
    for tv, sd in zip(noise.times, noise.sd_pulse):
        sd_p[np.argmin(np.abs(t - tv))] = sd
    sd_c = np.zeros(t.size)
    if fit_pandc and noise.sd_chase is not None:
        for tv, sd in zip(noise.times, noise.sd_chase):
            sd_c[np.argmin(np.abs(t - tv))] = sd
    miss_p = np.zeros(t.size)
    for tv, mf in zip(noise.times, noise.missing_pulse):
        miss_p[np.argmin(np.abs(t - tv))] = mf
    miss_c = np.zeros(t.size)
    if fit_pandc and noise.missing_chase is not None:
        for tv, mf in zip(noise.times, noise.missing_chase):
            sd_idx = np.argmin(np.abs(t - tv))
            miss_c[sd_idx] = mf

    rng = np.random.default_rng(seed)
    rows = []
    raw = {}
    hls = [float(h) for h in true_halflives]
    for hi, h in enumerate(hls):
        param_true = float(models.halflife_to_param(h, model))
        clean_p = models.predict(param_true, t, tpoint, chase=False, model=model, precursor=prec)
        clean_c = (models.predict(param_true, t, tpoint, chase=True, model=model, precursor=prec)
                   if fit_pandc else None)

        # (n_rep, T) noisy synthetic data
        synth_p = np.tile(clean_p, (n_rep, 1))
        noise_p = rng.normal(0.0, 1.0, size=(n_rep, meas_idx.size)) * (sd_p[meas_idx] * noise_scale)
        synth_p[:, meas_idx] = clean_p[meas_idx] + noise_p
        synth_p[:, 0] = 0.0
        # missingness
        pm = np.clip(miss_p[meas_idx] + extra_missing, 0.0, 0.95)
        drop = rng.random((n_rep, meas_idx.size)) < pm
        block = synth_p[:, meas_idx]
        block[drop] = np.nan
        synth_p[:, meas_idx] = block

        synth_c = None
        if fit_pandc:
            synth_c = np.tile(clean_c, (n_rep, 1))
            noise_c = rng.normal(0.0, 1.0, size=(n_rep, meas_idx.size)) * (sd_c[meas_idx] * noise_scale)
            synth_c[:, meas_idx] = clean_c[meas_idx] + noise_c
            synth_c[:, 0] = 0.0
            pmc = np.clip(miss_c[meas_idx] + extra_missing, 0.0, 0.95)
            dropc = rng.random((n_rep, meas_idx.size)) < pmc
            blockc = synth_c[:, meas_idx]
            blockc[dropc] = np.nan
            synth_c[:, meas_idx] = blockc

        res = fitting.fit_batch(t, synth_p, synth_c, tpoint, model, prec,
                                fit_pandc=fit_pandc, labels=None)
        est = np.array([r.half_life for r in res], dtype=float)
        clo = np.array([r.ci_low for r in res], dtype=float)
        chi = np.array([r.ci_high for r in res], dtype=float)
        note = np.array([r.note for r in res], dtype=object)

        rows.append(_metrics_row(h, est, clo, chi, note))
        raw[h] = {"est": est, "ci_low": clo, "ci_high": chi}
        if progress is not None:
            progress(hi + 1, len(hls))

    table = pd.DataFrame(rows)
    return SimulationResult(table=table, raw=raw, noise=noise, model=model,
                            fit_mode=("pulse+chase" if fit_pandc else "pulse"),
                            times=tuple(float(x) for x in t[meas]), n_rep=n_rep,
                            noise_scale=noise_scale, extra_missing=extra_missing, seed=seed)


def _metrics_row(h, est, clo, chi, note):
    fin = np.isfinite(est)
    e = est[fin]
    n = e.size
    med = float(np.median(e)) if n else np.nan
    mean = float(np.mean(e)) if n else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_err = float(np.median(np.abs(e - h) / h)) if n else np.nan
        rel_bias = (med - h) / h if n else np.nan
    # CI coverage over replicates with a finite CI
    ci_ok = np.isfinite(clo) & np.isfinite(chi)
    cov = float(np.mean((clo[ci_ok] <= h) & (h <= chi[ci_ok]))) if ci_ok.any() else np.nan
    ciw = chi - clo
    ciw = ciw[np.isfinite(ciw)]
    ci_med = float(np.median(ciw)) if ciw.size else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_ci = float(np.median((chi[ci_ok] - clo[ci_ok]) / est[ci_ok])) if ci_ok.any() else np.nan
    at_bound = float(np.mean(np.char.startswith(note.astype(str), "half-life at")))
    return {
        "TrueHalfLife_days": h,
        "N_rep": int(est.size),
        "N_finite": int(n),
        "Median_est_days": med,
        "Mean_est_days": mean,
        "Bias_days": (med - h) if n else np.nan,
        "PctBias": 100.0 * rel_bias if n else np.nan,
        "RelError_median": rel_err,
        "Coverage95": cov,
        "CI95_width_median_days": ci_med,
        "RelCI_median": rel_ci,
        "FracFinite": float(np.mean(fin)),
        "FracAtBound": at_bound,
    }


# --------------------------------------------------------------------------- #
#  turn the curves into a recommended reliable range
# --------------------------------------------------------------------------- #
def recommend_cutoff(sim: SimulationResult, pct_bias_tol=15.0, coverage_min=0.90,
                     rel_error_tol=0.50) -> dict:
    """Derive a data-driven reliable-recovery range from the simulation.

    A true half-life "passes" when |percent bias| <= pct_bias_tol AND coverage
    >= coverage_min AND median relative error <= rel_error_tol.  The reliable
    range is the longest contiguous block of passing half-lives (ties broken in
    favour of the block reaching the longer half-lives), so very short
    half-lives that are not resolvable by the sampling design do not hide an
    otherwise valid range.

    Returns a dict with the lower / upper limit of the range (``lower_days`` /
    ``cutoff_days``), the nearest failing values on either side, whether
    precision degrades gradually above the range, and EN/JP summary sentences.
    """
    df = sim.table.sort_values("TrueHalfLife_days").reset_index(drop=True)
    passed = (
        (df["PctBias"].abs() <= pct_bias_tol)
        & (df["Coverage95"] >= coverage_min)
        & (df["RelError_median"] <= rel_error_tol)
    ).to_numpy()
    hls = df["TrueHalfLife_days"].to_numpy(dtype=float)

    # contiguous blocks of passing values
    blocks, start = [], None
    for i, ok in enumerate(passed):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            blocks.append((start, i - 1)); start = None
    if start is not None:
        blocks.append((start, len(passed) - 1))

    criteria = {"pct_bias_tol": pct_bias_tol, "coverage_min": coverage_min,
                "rel_error_tol": rel_error_tol}
    if not blocks:
        return {
            "cutoff_days": None, "lower_days": None, "all_pass": False,
            "first_fail_days": float(hls[0]) if hls.size else None,
            "below_fail_days": None, "gradual_degradation": False,
            "criteria": criteria,
            "headline_en": "No tested half-life met all recovery criteria.",
            "headline_jp": "検証したいずれの半減期も回復基準を満たしませんでした。",
            "manuscript_en": ("Under the observed measurement noise the sampling design did not "
                              "recover half-lives within the predefined tolerance over the tested range."),
        }

    i0, i1 = max(blocks, key=lambda b: (b[1] - b[0], b[1]))
    lower, upper = float(hls[i0]), float(hls[i1])
    all_pass = bool(passed.all())
    below_fail = float(hls[i0 - 1]) if i0 > 0 else None
    first_fail = float(hls[i1 + 1]) if i1 + 1 < hls.size else None

    # gradual vs cliff above the lower limit: relative error rising monotonically
    # (assessed from the most precise tested half-life upward, so the poorer
    #  precision of very short half-lives does not mask the upper-end trend)
    re = df["RelError_median"].to_numpy(dtype=float)[i0:]
    re = re[np.isfinite(re)]
    if re.size:
        re = re[int(np.argmin(re)):]
    gradual = bool(re.size > 1 and np.all(np.diff(re) >= -1e-9))
    soft = "~" if (gradual and first_fail is not None) else ""
    approx = "approximately " if (gradual and first_fail is not None) else ""

    crit_en = (f"|bias| <= {pct_bias_tol:g}%, coverage >= {coverage_min:.0%}, "
               f"relative error <= {rel_error_tol:.0%}")
    crit_jp = (f"|bias| <= {pct_bias_tol:g}%、被覆確率 >= {coverage_min:.0%}、"
               f"相対誤差 <= {rel_error_tol:.0%}")

    if all_pass:
        headline = (f"All tested half-lives ({lower:g}-{upper:g} days) were recovered within "
                    f"tolerance ({crit_en}).")
        jp = f"検証したすべての半減期（{lower:g}〜{upper:g} 日）が許容範囲内で回復されました（{crit_jp}）。"
        manuscript = (f"Under the observed measurement noise, half-lives from {lower:g} to {upper:g} days "
                      f"were recovered with low bias and acceptable precision across the tested range.")
    else:
        headline = f"Half-lives from {lower:g} to {soft}{upper:g} days were recovered within tolerance ({crit_en})."
        jp = f"{lower:g}〜{soft}{upper:g} 日の半減期は許容範囲内で回復されました（{crit_jp}）。"
        tail_en, tail_jp = [], []
        if below_fail is not None:
            tail_en.append(f"half-lives of {below_fail:g} days or shorter were not resolved with "
                           f"acceptable precision by the sampling design")
            tail_jp.append(f"{below_fail:g} 日以下の短い半減期は十分な精度で分離できず")
        if first_fail is not None:
            tail_en.append(f"from {first_fail:g} days the confidence intervals widened and "
                           f"coverage/precision fell outside tolerance")
            tail_jp.append(f"{first_fail:g} 日以降は信頼区間が広がり被覆確率・精度が許容範囲を外れました")
        if tail_en:
            headline += " Outside this range, " + "; ".join(tail_en) + "."
            jp += " 範囲外では、" + "、".join(tail_jp) + "。"
        manuscript = (f"Under the observed measurement noise, half-lives from {lower:g} to "
                      f"{approx}{upper:g} days were recovered with low bias and acceptable precision")
        if first_fail is not None:
            manuscript += (", whereas estimates above this range showed progressively wider "
                           "confidence intervals")
        if below_fail is not None:
            manuscript += (f"; half-lives of {below_fail:g} days or shorter were poorly resolved "
                           f"by the sampling schedule")
        manuscript += "."

    return {
        "cutoff_days": upper,
        "lower_days": lower,
        "all_pass": all_pass,
        "first_fail_days": first_fail,
        "below_fail_days": below_fail,
        "gradual_degradation": gradual,
        "criteria": criteria,
        "headline_en": headline,
        "headline_jp": jp,
        "manuscript_en": manuscript,
    }
