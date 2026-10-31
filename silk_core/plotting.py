"""
Matplotlib figures for the SILK app (no interactive-plot dependency required).
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

from . import models, precursor as precursor_mod    # noqa: E402


def precursor_figure(result):
    """Precursor (Hsol) data + fitted/estimated bi-exponential curve(s).

    In 'plasma' mode the plasma points are the fit target; in 'global' mode the
    curve is estimated from all proteins and the plasma points (if present) are
    shown as an independent validation overlay.
    """
    prm = result.precursor_params
    is_global = prm.get("method") == "global"
    fig, ax = plt.subplots(figsize=(7, 4))

    plasma_lbl = "Plasma (validation)" if is_global else "Hsol Pulse (obs)"
    curve_lbl = "Global fit (from proteins)" if is_global else "Pulse fit"
    span = 20.0

    pp = result.precursor_pulse
    if pp is not None:
        ax.scatter(pp.time, pp.ratio, color="#1f77b4", s=30, label=plasma_lbl, zorder=3)
        span = float(np.nanmax(pp.time))

    pc = result.precursor_pandc
    if pc is not None and not is_global:
        ax.scatter(pc.time, pc.ratio, color="#d62728", s=30, marker="s",
                   label="Hsol Pulse&Chase (obs)", zorder=3)

    if np.isfinite(prm.get("T1", np.nan)):
        tt = np.linspace(0, span, 300)
        ax.plot(tt, precursor_mod.precursor_curve(prm, tt, chase=False),
                color="#1f77b4", lw=2, label=curve_lbl)
        if not is_global and np.isfinite(prm.get("A1", np.nan)) and pc is not None:
            ax.plot(tt, precursor_mod.precursor_curve(prm, tt, chase=True),
                    color="#d62728", lw=2, ls="--", label="Pulse&Chase fit")

    if prm.get("tpoint") is not None:
        ax.axvline(prm["tpoint"], color="gray", ls=":", lw=1,
                   label=f"chase switch = {prm['tpoint']:g} d")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Label / Total ratio")
    title = ("Precursor estimated from all proteins (paper method)"
             if is_global else "Soluble precursor (Hsol) labelling — plasma fit")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def halflife_histogram(table, log_scale=True, max_days=None):
    """Distribution of fitted half-lives."""
    hl = table["HalfLife_days"].to_numpy(dtype=float)
    hl = hl[np.isfinite(hl) & (hl > 0)]
    if max_days:
        hl = hl[hl <= max_days]
    fig, ax = plt.subplots(figsize=(7, 4))
    if hl.size == 0:
        ax.text(0.5, 0.5, "No finite half-lives", ha="center", va="center")
        return fig
    if log_scale:
        bins = np.logspace(np.log10(max(hl.min(), 1e-2)), np.log10(hl.max()), 40)
        ax.hist(hl, bins=bins, color="#2ca02c", alpha=0.8, edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Half-life (days, log scale)")
    else:
        ax.hist(hl, bins=40, color="#2ca02c", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Half-life (days)")
    ax.axvline(np.median(hl), color="black", ls="--", lw=1.5,
               label=f"median = {np.median(hl):.1f} d")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Half-life distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def lifetime_curve_figure(result, x_range=(0.0, 60.0), y_range=None,
                          timepoints=None, in_percent=True,
                          show_pulse=True, show_chase=True,
                          show_points=True, max_scatter=5000):
    """
    Label-incorporation vs. lifetime plot, in the style of Fornasiero et al.
    2018, Supplementary Fig. 3i.

    For each pulse time point the model predicts the Label/Total ratio as a
    function of protein lifetime (a decreasing curve: longer-lived proteins
    incorporate less label at a fixed time).  Every protein is then scattered at
    (its fitted half-life, its measured ratio) for each time point, so well-fit
    proteins fall on the corresponding curve.

    x_range / y_range : adjustable axis limits. y_range is in the same units as
    the y-axis (percent if in_percent, else fraction). Pass None for y auto.
    """
    t = np.asarray(result.protein_pulse.time, dtype=float)     # e.g. [0,5,14,21]
    meas_times = t[t > 0]
    if timepoints is None:
        timepoints = list(meas_times)
    tpoint = result.tpoint
    model = result.model
    prec = result.precursor_params
    scale = 100.0 if in_percent else 1.0

    xmin, xmax = float(x_range[0]), float(x_range[1])
    if xmax <= xmin:
        xmax = xmin + 1.0
    L = np.linspace(max(xmin, 1e-3), xmax, 400)                # lifetime axis
    params = np.asarray(models.halflife_to_param(L, model), dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    palette = plt.cm.viridis(np.linspace(0.12, 0.82, max(len(meas_times), 1)))
    tcol = {float(tv): palette[j] for j, tv in enumerate(meas_times)}
    last_t = float(meas_times[-1]) if meas_times.size else tpoint

    hl = result.table["HalfLife_days"].to_numpy(dtype=float)
    ratios = result.protein_pulse.ratios

    def _scatter(xs, ys, color, marker, label):
        m = np.isfinite(xs) & np.isfinite(ys) & (xs >= xmin) & (xs <= xmax)
        xs, ys = xs[m], ys[m] * scale
        if xs.size > max_scatter:
            step = int(np.ceil(xs.size / max_scatter))
            xs, ys = xs[::step], ys[::step]
        ax.scatter(xs, ys, s=7, alpha=0.35, color=color, marker=marker,
                   edgecolors="none", label=label, zorder=2)

    # ---- pulse: model curve + data for each requested time point ----
    if show_pulse:
        for tv in timepoints:
            tv = float(tv)
            pred = models.predict_grid(params, np.array([tv]), tpoint,
                                       chase=False, model=model, precursor=prec)[:, 0]
            ax.plot(L, pred * scale, color=tcol.get(tv, "gray"), lw=2.2, zorder=3,
                    label=f"{tv:g} d pulse — model")
            if show_points:
                col = int(np.argmin(np.abs(t - tv)))
                _scatter(hl, ratios[:, col], tcol.get(tv, "gray"), "o",
                         f"{tv:g} d pulse — data")

    # ---- pulse&chase: model curve + data at the last time point ----
    if show_chase and result.pandc_aligned is not None:
        predc = models.predict_grid(params, np.array([last_t]), tpoint,
                                    chase=True, model=model, precursor=prec)[:, 0]
        ax.plot(L, predc * scale, color="#d62728", lw=2.2, ls="--", zorder=3,
                label=f"{last_t:g} d pulse&chase — model")
        if show_points:
            col = int(np.argmin(np.abs(t - last_t)))
            _scatter(hl, result.pandc_aligned[:, col], "#d62728", "s",
                     f"{last_t:g} d chase — data")

    ax.set_xlim(xmin, xmax)
    if y_range is not None:
        ax.set_ylim(float(y_range[0]), float(y_range[1]))
    ax.set_xlabel("Protein half-life (days)")
    ax.set_ylabel(r"$^{13}$C$_6$,$^{15}$N$_2$-lysine incorporation (%)" if in_percent
                  else "Label / Total ratio")
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def protein_fit_figure(result, row_index):
    """Observed points + fitted labelling curve for a single protein."""
    fr = result.fit_results[row_index]
    t = result.protein_pulse.time
    obs_p = result.protein_pulse.ratios[row_index]
    tpoint = result.tpoint
    model = result.model
    prec = result.precursor_params

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(t, obs_p, color="#1f77b4", s=45, zorder=3, label="Pulse (obs)")

    obs_c = None
    if getattr(result, "pandc_aligned", None) is not None and row_index < result.pandc_aligned.shape[0]:
        obs_c = result.pandc_aligned[row_index]
        if np.isfinite(obs_c).any():
            ax.scatter(t, obs_c, color="#d62728", s=45, marker="s", zorder=3,
                       label="Pulse&Chase (obs)")

    if np.isfinite(fr.parameter):
        tt = np.linspace(0, float(np.nanmax(t)), 400)
        ax.plot(tt, models.predict(fr.parameter, tt, tpoint, chase=False,
                                    model=model, precursor=prec),
                color="#1f77b4", lw=2, label="Pulse fit")
        if result.fit_mode == "pulse+chase" and obs_c is not None:
            ax.plot(tt, models.predict(fr.parameter, tt, tpoint, chase=True,
                                       model=model, precursor=prec),
                    color="#d62728", lw=2, ls="--", label="Pulse&Chase fit")

    ax.axvline(tpoint, color="gray", ls=":", lw=1)
    title = fr.gene or fr.protein_group or f"protein #{row_index}"
    hl = fr.half_life
    ax.set_title(f"{title}   half-life = {hl:.1f} d" if np.isfinite(hl) else title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Label / Total ratio")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  fit-quality (QC) figures  -- added feature
# --------------------------------------------------------------------------- #
def quality_scatter_figure(qdf, rel_ci_max=0.30, rel_ci_basis="halfwidth",
                           cutoff_days=None, max_points=6000):
    """Per-protein precision vs half-life, coloured by QC pass/fail.

    Left panel : relative 95 % CI (y) against fitted half-life (x, log).  The
    dashed line is the acceptance threshold; points below it (green) pass, points
    above (grey) are filtered out.  An optional vertical line marks a
    simulation-derived reliable-recovery cutoff.
    Right panel: half-life distribution split into retained vs filtered.
    """
    col = "RelCI_halfwidth" if rel_ci_basis == "halfwidth" else "RelCI_width"
    hl = qdf["HalfLife_days"].to_numpy(dtype=float)
    rc = qdf[col].to_numpy(dtype=float)
    passed = qdf["QC_pass"].to_numpy(dtype=bool) if "QC_pass" in qdf else (rc <= rel_ci_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    m = np.isfinite(hl) & np.isfinite(rc) & (hl > 0)
    hlm, rcm, pm = hl[m], rc[m], passed[m]
    # subsample for a readable scatter
    if hlm.size > max_points:
        step = int(np.ceil(hlm.size / max_points))
        hlm, rcm, pm = hlm[::step], rcm[::step], pm[::step]
    ax1.scatter(hlm[~pm], np.clip(rcm[~pm], 0, 2), s=8, alpha=0.3, color="#9e9e9e",
                edgecolors="none", label="filtered out")
    ax1.scatter(hlm[pm], np.clip(rcm[pm], 0, 2), s=9, alpha=0.5, color="#2ca02c",
                edgecolors="none", label="retained (QC pass)")
    ax1.axhline(rel_ci_max, color="#d62728", ls="--", lw=1.5,
                label=f"threshold = {rel_ci_max:g}")
    if cutoff_days:
        ax1.axvline(cutoff_days, color="#1f77b4", ls=":", lw=1.5,
                    label=f"sim. cutoff = {cutoff_days:g} d")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 2)
    ax1.set_xlabel("Fitted half-life (days, log)")
    ylab = "relative 95% CI ( +/- fraction )" if rel_ci_basis == "halfwidth" else "relative 95% CI (width/est)"
    ax1.set_ylabel(ylab)
    ax1.set_title("Per-protein precision vs half-life")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    pos = hlm[hlm > 0]
    if pos.size:
        bins = np.logspace(np.log10(max(pos.min(), 1e-2)), np.log10(pos.max()), 40)
        ax2.hist(hlm[~pm][hlm[~pm] > 0], bins=bins, color="#9e9e9e", alpha=0.7,
                 label="filtered out")
        ax2.hist(hlm[pm][hlm[pm] > 0], bins=bins, color="#2ca02c", alpha=0.8,
                 label="retained")
        ax2.set_xscale("log")
    if cutoff_days:
        ax2.axvline(cutoff_days, color="#1f77b4", ls=":", lw=1.5)
    ax2.set_xlabel("Half-life (days, log)")
    ax2.set_ylabel("Number of proteins")
    ax2.set_title("Retained vs filtered half-lives")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  simulation figures  -- added feature
# --------------------------------------------------------------------------- #
def simulation_figure(sim, cutoff_days=None, pct_bias_tol=15.0,
                      coverage_min=0.90, rel_error_tol=0.50, show_header=True):
    """Four-panel recovery summary for a :class:`simulation.SimulationResult`.

    (a) recovered vs true half-life (median +/- inter-quartile range, y=x line),
    (b) percent bias vs true half-life (+/- tolerance band),
    (c) 95 % CI coverage vs true half-life (0.95 target, >= coverage_min band),
    (d) median relative error vs true half-life (descriptive; curve only).
    ``show_header=False`` omits the figure header (publication style).
    A vertical line marks the data-driven reliable-recovery cutoff.
    """
    df = sim.table.sort_values("TrueHalfLife_days").reset_index(drop=True)
    x = df["TrueHalfLife_days"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axa, axb, axc, axd = axes.ravel()

    # (a) recovered vs true
    med = df["Median_est_days"].to_numpy(dtype=float)
    lo = np.array([np.nanpercentile(sim.raw[h]["est"], 25) for h in x])
    hi = np.array([np.nanpercentile(sim.raw[h]["est"], 75) for h in x])
    axa.plot([x.min(), x.max()], [x.min(), x.max()], color="gray", ls="--", lw=1, label="y = x (unbiased)")
    axa.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o-", color="#1f77b4",
                 capsize=4, lw=1.8, label="median estimate (IQR)")
    axa.set_xlabel("True half-life (days)")
    axa.set_ylabel("Recovered half-life (days)")
    axa.set_title("(a) Recovery: estimated vs true")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.3)

    # (b) percent bias
    axb.axhspan(-pct_bias_tol, pct_bias_tol, color="#2ca02c", alpha=0.12,
                label=f"+/-{pct_bias_tol:g}% band")
    axb.axhline(0, color="gray", lw=1)
    axb.plot(x, df["PctBias"], "o-", color="#d62728", lw=1.8)
    axb.set_xlabel("True half-life (days)")
    axb.set_ylabel("Median bias (%)")
    axb.set_title("(b) Bias vs true half-life")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.3)

    # (c) coverage
    axc.axhspan(coverage_min, 1.0, color="#2ca02c", alpha=0.12,
                label=f">= {coverage_min:.0%} acceptable")
    axc.axhline(0.95, color="gray", ls="--", lw=1, label="0.95 nominal")
    axc.plot(x, df["Coverage95"], "o-", color="#9467bd", lw=1.8)
    axc.set_ylim(0.5, 1.02)
    axc.set_xlabel("True half-life (days)")
    axc.set_ylabel("95% CI coverage")
    axc.set_title("(c) CI coverage vs true half-life")
    axc.legend(fontsize=8)
    axc.grid(alpha=0.3)

    # (d) median relative error -- descriptive precision measure (curve only)
    axd.plot(x, df["RelError_median"], "o-", color="#ff7f0e", lw=1.8)
    axd.set_ylim(bottom=0)
    axd.set_xlabel("True half-life (days)")
    axd.set_ylabel("median |estimate - true| / true")
    axd.set_title("(d) Estimation accuracy vs true half-life")
    axd.grid(alpha=0.3)

    if x.size and np.nanmin(x) > 0 and np.nanmax(x) / np.nanmin(x) > 50:
        from matplotlib.ticker import FuncFormatter
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        for ax in (axa, axb, axc, axd):
            ax.set_xscale("log")
            ax.set_xticks(x)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.tick_params(axis="x", labelsize=7, rotation=45)
        axa.set_yscale("log")
        axa.yaxis.set_major_formatter(fmt)
        axa.set_xlabel("True half-life (days, log)")
        axb.set_xlabel("True half-life (days, log)")
        axc.set_xlabel("True half-life (days, log)")
        axd.set_xlabel("True half-life (days, log)")

    if show_header:
        fig.suptitle(f"Half-life recovery under measured noise "
                     f"(model={sim.model}, n={sim.n_rep}/point, sampling={', '.join(f'{v:g}' for v in sim.times)} d)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        # publication style: no header (the description goes in the figure legend)
        fig.tight_layout()
    return fig


def figure_to_bytes(fig, fmt="png", dpi=300) -> bytes:
    """Serialise a matplotlib figure (png / pdf / svg / tiff) for download."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
