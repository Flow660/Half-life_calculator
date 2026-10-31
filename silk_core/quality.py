"""
Per-protein fit-quality assessment and a goodness-of-fit (GOF) filter.

This module adds a *quality layer* on top of an existing pipeline run
(:class:`silk_core.pipeline.RunResult`).  It does **not** change how the
half-lives themselves are fitted; it summarises how trustworthy each fit is and
lets the user keep only the proteins whose fit meets predefined criteria.

For every protein it reports:

* ``R2``                      -- coefficient of determination of the fit.
* ``RMSE`` / ``WeightedResid``-- root-mean-square residual over the fitted points
                                 (the residual the objective function minimises).
* ``HalfLife_SE_days``        -- asymptotic standard error (delta method).
* ``CI95_low/high_days``      -- asymptotic 95 % confidence interval.
* ``CI95_width_days``         -- ci_high - ci_low.
* ``RelCI_halfwidth``         -- (ci_high - ci_low) / 2 / half_life, i.e. the CI
                                 expressed as a +/- fraction of the estimate.
* ``RelCI_width``             -- (ci_high - ci_low) / half_life (full width).
* ``CV``                      -- SE / half_life (coefficient of variation).
* ``N_timepoints`` / ``N_observed`` / ``MissingFrac`` -- per-protein sampling
                                 completeness across the design time points.
* ``N_peptides`` / ``N_replicates`` -- carried through **iff** the caller supplies
                                 per-protein peptide / replicate counts (the four
                                 standard SILK Excel files store one Label/Total
                                 ratio per protein per time point, so these are
                                 ``NaN`` unless a metadata table is provided).

The filter :func:`apply_gof_filter` marks each protein ``QC_pass`` True/False and
records the reason it failed, using thresholds that default to the criteria a
reviewer would expect (e.g. "the 95 % CI was below 30 % of the estimated
half-life").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import models


# --------------------------------------------------------------------------- #
#  dataset-level missingness
# --------------------------------------------------------------------------- #
def per_timepoint_missing(protein_data, drop_zero=True) -> pd.DataFrame:
    """Fraction of proteins with a missing (non-finite) ratio at each time point.

    Returns a DataFrame with columns ``Time_days``, ``N_total``, ``N_observed``,
    ``MissingFrac``.
    """
    t = np.asarray(protein_data.time, dtype=float)
    R = np.asarray(protein_data.ratios, dtype=float)
    cols = np.arange(t.size)
    if drop_zero:
        cols = cols[t > 0]
    rows = []
    n_total = R.shape[0]
    for c in cols:
        finite = int(np.isfinite(R[:, c]).sum())
        rows.append({
            "Time_days": float(t[c]),
            "N_total": n_total,
            "N_observed": finite,
            "MissingFrac": float(1.0 - finite / n_total) if n_total else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  per-protein quality table
# --------------------------------------------------------------------------- #
def compute_quality_table(result, peptide_counts=None, replicate_counts=None) -> pd.DataFrame:
    """Build a per-protein quality table from a completed run.

    Parameters
    ----------
    result : silk_core.pipeline.RunResult
    peptide_counts / replicate_counts : optional length-N arrays (or dicts keyed
        by ProteinGroups) giving the number of quantified peptides / biological
        replicates behind each protein.  When omitted the corresponding columns
        are NaN, because the standard four-file SILK format carries a single
        ratio per protein per time point.

    Returns
    -------
    pandas.DataFrame -- one row per protein, aligned to ``result.table``.
    """
    fr = result.fit_results
    n = len(fr)
    t = np.asarray(result.protein_pulse.time, dtype=float)
    meas = t > 0
    n_design_pulse = int(meas.sum())

    pulse = np.asarray(result.protein_pulse.ratios, dtype=float)
    pandc = getattr(result, "pandc_aligned", None)
    fit_pandc = result.fit_mode == "pulse+chase" and pandc is not None

    pg = [f.protein_group for f in fr]
    pep = _coerce_counts(peptide_counts, pg, n)
    rep = _coerce_counts(replicate_counts, pg, n)

    rows = []
    for i, f in enumerate(fr):
        hl = f.half_life
        ciw = f.ci_high - f.ci_low
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_hw = (ciw / 2.0) / hl if np.isfinite(hl) and hl != 0 else np.nan
            rel_w = ciw / hl if np.isfinite(hl) and hl != 0 else np.nan
            cv = f.half_life_se / hl if np.isfinite(hl) and hl != 0 else np.nan
        rmse = np.sqrt(f.sse / f.n_points) if f.n_points and np.isfinite(f.sse) else np.nan

        # per-protein sampling completeness
        n_obs = int(np.isfinite(pulse[i, meas]).sum())
        n_design = n_design_pulse
        if fit_pandc and i < pandc.shape[0]:
            n_obs += int(np.isfinite(pandc[i, meas]).sum())
            n_design += n_design_pulse
        miss = 1.0 - n_obs / n_design if n_design else np.nan

        rows.append({
            "index": f.index,
            "ProteinGroups": f.protein_group,
            "Gene": f.gene,
            "Description": f.description,
            "HalfLife_days": hl,
            "HalfLife_SE_days": f.half_life_se,
            "CI95_low_days": f.ci_low,
            "CI95_high_days": f.ci_high,
            "CI95_width_days": ciw if np.isfinite(ciw) else np.nan,
            "RelCI_halfwidth": rel_hw,
            "RelCI_width": rel_w,
            "CV": cv,
            "R2": f.r2,
            "RMSE": rmse,
            "SSE": f.sse,
            "N_timepoints": n_design,
            "N_observed": n_obs,
            "MissingFrac": miss,
            "N_peptides": pep[i],
            "N_replicates": rep[i],
            "AtBound": bool(f.note.startswith("half-life at")),
            "Note": f.note,
        })
    return pd.DataFrame(rows)


def _coerce_counts(counts, protein_groups, n):
    out = np.full(n, np.nan)
    if counts is None:
        return out
    if isinstance(counts, dict):
        for i, pg in enumerate(protein_groups):
            if pg in counts:
                out[i] = counts[pg]
        return out
    arr = np.asarray(counts, dtype=float).ravel()
    m = min(n, arr.size)
    out[:m] = arr[:m]
    return out


# --------------------------------------------------------------------------- #
#  goodness-of-fit filter
# --------------------------------------------------------------------------- #
@dataclass
class GOFThresholds:
    """Predefined goodness-of-fit criteria for retaining a half-life estimate."""
    r2_min: float = 0.90
    rel_ci_max: float = 0.50            # applied to RelCI_halfwidth (CI as +/- fraction)
    rel_ci_basis: str = "halfwidth"     # 'halfwidth' or 'width'
    min_observed: int = 3
    max_missing_frac: float = 0.5
    exclude_at_bound: bool = True
    require_finite_ci: bool = True
    max_halflife_days: float | None = None   # optional hard ceiling on the estimate

    def describe(self) -> str:
        basis = ("half-width, i.e. +/-fraction of the estimate"
                 if self.rel_ci_basis == "halfwidth" else "full width / estimate")
        parts = [
            f"R2 >= {self.r2_min:g}",
            f"relative 95% CI ({basis}) <= {self.rel_ci_max:g}",
            f">= {self.min_observed} observed points",
            f"missing fraction <= {self.max_missing_frac:g}",
        ]
        if self.exclude_at_bound:
            parts.append("estimate not pinned to a fit bound")
        if self.require_finite_ci:
            parts.append("finite CI")
        if self.max_halflife_days is not None:
            parts.append(f"half-life <= {self.max_halflife_days:g} d")
        return "; ".join(parts)


def apply_gof_filter(qdf: pd.DataFrame, thresholds: GOFThresholds | None = None) -> pd.DataFrame:
    """Add ``QC_pass`` (bool) and ``QC_reason`` (str) columns.

    A protein passes when it satisfies every criterion in ``thresholds``.
    ``QC_reason`` lists the criteria it violates (empty string when it passes).
    """
    th = thresholds or GOFThresholds()
    df = qdf.copy()

    rel_ci = df["RelCI_halfwidth"] if th.rel_ci_basis == "halfwidth" else df["RelCI_width"]

    reasons = []
    passes = []
    for i in range(len(df)):
        r = df.iloc[i]
        why = []
        if not (np.isfinite(r["R2"]) and r["R2"] >= th.r2_min):
            why.append(f"R2<{th.r2_min:g}")
        rc = rel_ci.iloc[i]
        if not (np.isfinite(rc) and rc <= th.rel_ci_max):
            why.append("CI too wide" if np.isfinite(rc) else "CI not estimable")
        if not (np.isfinite(r["N_observed"]) and r["N_observed"] >= th.min_observed):
            why.append("too few points")
        if not (np.isfinite(r["MissingFrac"]) and r["MissingFrac"] <= th.max_missing_frac):
            why.append("too much missing")
        if th.exclude_at_bound and bool(r["AtBound"]):
            why.append("at fit bound")
        if th.require_finite_ci and not (np.isfinite(r["CI95_low_days"]) and np.isfinite(r["CI95_high_days"])):
            why.append("CI not finite")
        if th.max_halflife_days is not None:
            if not (np.isfinite(r["HalfLife_days"]) and r["HalfLife_days"] <= th.max_halflife_days):
                why.append(f"half-life>{th.max_halflife_days:g}d")
        passes.append(len(why) == 0)
        reasons.append("; ".join(why))
    df["QC_pass"] = passes
    df["QC_reason"] = reasons
    return df


def qc_summary(qdf_filtered: pd.DataFrame) -> dict:
    """Counts for a filtered quality table (expects a ``QC_pass`` column)."""
    n = len(qdf_filtered)
    n_pass = int(qdf_filtered["QC_pass"].sum()) if n else 0
    hl_pass = qdf_filtered.loc[qdf_filtered["QC_pass"], "HalfLife_days"]
    hl_pass = hl_pass[np.isfinite(hl_pass)]
    return {
        "n_total": n,
        "n_pass": n_pass,
        "n_fail": n - n_pass,
        "pass_frac": (n_pass / n) if n else np.nan,
        "median_hl_pass": float(np.median(hl_pass)) if hl_pass.size else np.nan,
        "max_hl_pass": float(np.max(hl_pass)) if hl_pass.size else np.nan,
    }


# --------------------------------------------------------------------------- #
#  bootstrap confidence interval (residual resampling)
# --------------------------------------------------------------------------- #
def bootstrap_ci_batch(result, indices=None, n_boot=300, seed=0, ci=95.0):
    """Residual-bootstrap 95 % CI for the half-life of selected proteins.

    For each protein the fitted prediction is held fixed and its residuals are
    resampled with replacement; the perturbed curve is re-fitted and the
    distribution of recovered half-lives gives a percentile CI.  This is the
    non-asymptotic complement to the delta-method CI in the results table.

    All bootstrap replicates of all requested proteins for a given draw are fit
    in a single vectorised :func:`fitting.fit_batch` call, so a few hundred
    proteins take a second or two.

    Parameters
    ----------
    indices : iterable of protein row indices, or None for all proteins.
    n_boot  : bootstrap replicates.
    seed    : RNG seed (reproducible).

    Returns
    -------
    pandas.DataFrame with columns index, ProteinGroups, Gene,
    Boot_CI95_low_days, Boot_CI95_high_days, Boot_SE_days, Boot_median_days.
    """
    from . import fitting

    fr = result.fit_results
    if indices is None:
        indices = range(len(fr))
    indices = [int(i) for i in indices]
    if not indices:
        return pd.DataFrame(columns=["index", "ProteinGroups", "Gene",
                                     "Boot_CI95_low_days", "Boot_CI95_high_days",
                                     "Boot_SE_days", "Boot_median_days"])

    t = np.asarray(result.protein_pulse.time, dtype=float)
    tpoint = result.tpoint
    model = result.model
    prec = result.precursor_params
    fit_pandc = result.fit_mode == "pulse+chase" and getattr(result, "pandc_aligned", None) is not None

    pulse = np.asarray(result.protein_pulse.ratios, dtype=float)
    pandc = np.asarray(result.pandc_aligned, dtype=float) if fit_pandc else None

    rng = np.random.default_rng(seed)
    meas_idx = np.where(t > 0)[0]

    # fitted predictions for the selected proteins
    params = np.array([fr[i].parameter for i in indices], dtype=float)
    ok = np.isfinite(params)
    lo_q, hi_q = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0

    n_sel = len(indices)
    boot_hl = np.full((n_boot, n_sel), np.nan)

    # precompute per-protein clean predictions and residuals
    clean_p = np.full((n_sel, t.size), np.nan)
    resid_p = [None] * n_sel
    clean_c = np.full((n_sel, t.size), np.nan) if fit_pandc else None
    resid_c = [None] * n_sel if fit_pandc else None
    for j, i in enumerate(indices):
        if not ok[j]:
            continue
        pr = models.predict(params[j], t, tpoint, chase=False, model=model, precursor=prec)
        clean_p[j] = pr
        obs = pulse[i]
        m = np.isfinite(obs) & (t > 0)
        resid_p[j] = (obs[m] - pr[m])
        if fit_pandc:
            prc = models.predict(params[j], t, tpoint, chase=True, model=model, precursor=prec)
            clean_c[j] = prc
            oc = pandc[i]
            mc = np.isfinite(oc) & (t > 0)
            resid_c[j] = (oc[mc] - prc[mc])

    for b in range(n_boot):
        synth_p = np.full((n_sel, t.size), np.nan)
        synth_c = np.full((n_sel, t.size), np.nan) if fit_pandc else None
        for j in range(n_sel):
            if not ok[j]:
                continue
            synth_p[j] = clean_p[j]
            rp = resid_p[j]
            if rp is not None and rp.size:
                draw = rng.choice(rp, size=meas_idx.size, replace=True)
                synth_p[j, meas_idx] = clean_p[j, meas_idx] + draw
            synth_p[j, 0] = 0.0
            if fit_pandc:
                synth_c[j] = clean_c[j]
                rc = resid_c[j]
                if rc is not None and rc.size:
                    drawc = rng.choice(rc, size=meas_idx.size, replace=True)
                    synth_c[j, meas_idx] = clean_c[j, meas_idx] + drawc
                synth_c[j, 0] = 0.0
        res = fitting.fit_batch(t, synth_p, synth_c, tpoint, model, prec,
                                fit_pandc=fit_pandc, labels=None)
        boot_hl[b] = [r.half_life for r in res]

    out = []
    for j, i in enumerate(indices):
        col = boot_hl[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            lo, hi = np.percentile(col, [lo_q, hi_q])
            out.append({
                "index": i,
                "ProteinGroups": fr[i].protein_group,
                "Gene": fr[i].gene,
                "Boot_CI95_low_days": float(lo),
                "Boot_CI95_high_days": float(hi),
                "Boot_SE_days": float(np.std(col, ddof=1)),
                "Boot_median_days": float(np.median(col)),
            })
        else:
            out.append({
                "index": i, "ProteinGroups": fr[i].protein_group, "Gene": fr[i].gene,
                "Boot_CI95_low_days": np.nan, "Boot_CI95_high_days": np.nan,
                "Boot_SE_days": np.nan, "Boot_median_days": np.nan,
            })
    return pd.DataFrame(out)
