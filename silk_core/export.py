"""
Export results to Excel / CSV (in-memory buffers for download).
"""

from __future__ import annotations

import io

import pandas as pd


def results_to_excel_bytes(result) -> bytes:
    """Build an .xlsx with a Results sheet, a Precursor sheet and a Run-info sheet."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        result.table.to_excel(xl, sheet_name="Half-life results", index=False)

        prm = result.precursor_params
        prec_df = pd.DataFrame({
            "Parameter": ["T1 (days)", "T2 (days)", "A", "A1", "A2",
                          "Chase switch tpoint (days)", "R2 pulse", "R2 chase"],
            "Value": [prm.get("T1"), prm.get("T2"), prm.get("A"),
                      prm.get("A1"), prm.get("A2"), prm.get("tpoint"),
                      prm.get("r2_pulse"), prm.get("r2_chase")],
        })
        prec_df.to_excel(xl, sheet_name="Precursor fit", index=False)

        prm = result.precursor_params
        info = pd.DataFrame({
            "Field": ["Model", "Precursor source", "Precursor method",
                      "Fit mode", "Chase switch (days)",
                      "Proteins fitted", "Finite half-lives"],
            "Value": [
                result.model,
                getattr(result, "precursor_source", "plasma"),
                prm.get("method", "n/a"),
                result.fit_mode, result.tpoint,
                len(result.table),
                int(result.table["HalfLife_days"].notna().sum()),
            ],
        })
        info.to_excel(xl, sheet_name="Run info", index=False)

        # light formatting
        wb = xl.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        for sheet_name, df in [("Half-life results", result.table),
                               ("Precursor fit", prec_df), ("Run info", info)]:
            ws = xl.sheets[sheet_name]
            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)
                width = max(12, min(40, int(df[col].astype(str).str.len().max() if len(df) else 12) + 2))
                ws.set_column(c, c, width)
            ws.freeze_panes(1, 0)
    buf.seek(0)
    return buf.getvalue()


def results_to_csv_bytes(result) -> bytes:
    return result.table.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
#  fit-quality (QC) export  -- added feature
# --------------------------------------------------------------------------- #
def _autofit(xl, sheet_name, df, header_fmt):
    ws = xl.sheets[sheet_name]
    for c, col in enumerate(df.columns):
        ws.write(0, c, col, header_fmt)
        try:
            body_w = int(df[col].astype(str).str.len().max()) if len(df) else 12
        except Exception:
            body_w = 12
        ws.set_column(c, c, max(12, min(44, max(body_w, len(str(col))) + 2)))
    ws.freeze_panes(1, 0)


def quality_to_excel_bytes(result, qdf, thresholds, missing_df=None) -> bytes:
    """Excel workbook for the fit-quality / QC filter output.

    Sheets: QC results (per protein), Criteria, Missingness, Run info.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        qdf.to_excel(xl, sheet_name="QC results", index=False)

        crit = pd.DataFrame({
            "Criterion": ["R2 min", "Relative 95% CI max", "Relative CI basis",
                          "Min observed points", "Max missing fraction",
                          "Exclude at-bound estimates", "Require finite CI",
                          "Max half-life (days)"],
            "Value": [thresholds.r2_min, thresholds.rel_ci_max, thresholds.rel_ci_basis,
                      thresholds.min_observed, thresholds.max_missing_frac,
                      thresholds.exclude_at_bound, thresholds.require_finite_ci,
                      thresholds.max_halflife_days],
        })
        crit.to_excel(xl, sheet_name="Criteria", index=False)

        n_total = len(qdf)
        n_pass = int(qdf["QC_pass"].sum()) if "QC_pass" in qdf else 0
        summ = pd.DataFrame({
            "Field": ["Model", "Fit mode", "Proteins total", "Proteins retained (QC pass)",
                      "Proteins filtered out", "Retained fraction"],
            "Value": [result.model, result.fit_mode, n_total, n_pass, n_total - n_pass,
                      round(n_pass / n_total, 4) if n_total else None],
        })
        summ.to_excel(xl, sheet_name="Run info", index=False)

        sheets = [("QC results", qdf), ("Criteria", crit), ("Run info", summ)]
        if missing_df is not None:
            missing_df.to_excel(xl, sheet_name="Missingness", index=False)
            sheets.append(("Missingness", missing_df))

        wb = xl.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        for name, df in sheets:
            _autofit(xl, name, df, header_fmt)
    buf.seek(0)
    return buf.getvalue()


def quality_to_csv_bytes(qdf) -> bytes:
    return qdf.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
#  simulation export  -- added feature
# --------------------------------------------------------------------------- #
def simulation_to_excel_bytes(sim, recommendation=None) -> bytes:
    """Excel workbook for the recovery-simulation output.

    Sheets: Recovery metrics (per true half-life), Noise model, Settings,
    and (optional) Recommendation.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
        sim.table.to_excel(xl, sheet_name="Recovery metrics", index=False)

        nm = sim.noise
        noise_df = pd.DataFrame({
            "Time_days": list(nm.times),
            "NoiseSD_pulse": list(nm.sd_pulse),
            "NoiseSD_chase": list(nm.sd_chase) if nm.sd_chase is not None else [None] * len(nm.times),
            "MissingFrac_pulse": list(nm.missing_pulse),
            "MissingFrac_chase": list(nm.missing_chase) if nm.missing_chase is not None else [None] * len(nm.times),
        }) if nm is not None else pd.DataFrame()
        noise_df.to_excel(xl, sheet_name="Noise model", index=False)

        settings = pd.DataFrame({
            "Field": ["Model", "Fit mode", "Sampling times (days)", "Replicates per point",
                      "Noise scale", "Extra missing", "Pooled noise SD", "Noise estimated from",
                      "Proteins used for noise", "Proteins total", "RNG seed"],
            "Value": [sim.model, sim.fit_mode, ", ".join(f"{v:g}" for v in sim.times),
                      sim.n_rep, sim.noise_scale, sim.extra_missing,
                      round(nm.pooled_sd, 5) if nm else None,
                      (nm.selection_label() if hasattr(nm, "selection_label") else None) if nm else None,
                      nm.n_used if nm else None, getattr(nm, "n_total", None) if nm else None,
                      sim.seed],
        })
        settings.to_excel(xl, sheet_name="Settings", index=False)

        sheets = [("Recovery metrics", sim.table), ("Noise model", noise_df),
                  ("Settings", settings)]
        if recommendation is not None:
            rec_df = pd.DataFrame({
                "Field": ["Reliable range lower limit (days)", "Reliable range upper limit (days)",
                          "All tested passed", "First failing above range (days)",
                          "Nearest failing below range (days)",
                          "Gradual degradation", "Bias tolerance (%)", "Coverage min",
                          "Rel. error tol", "Summary (EN)", "Suggested manuscript sentence (EN)"],
                "Value": [recommendation.get("lower_days"), recommendation.get("cutoff_days"),
                          recommendation.get("all_pass"), recommendation.get("first_fail_days"),
                          recommendation.get("below_fail_days"), recommendation.get("gradual_degradation"),
                          recommendation["criteria"]["pct_bias_tol"],
                          recommendation["criteria"]["coverage_min"],
                          recommendation["criteria"]["rel_error_tol"],
                          recommendation.get("headline_en"), recommendation.get("manuscript_en")],
            })
            rec_df.to_excel(xl, sheet_name="Recommendation", index=False)
            sheets.append(("Recommendation", rec_df))

        wb = xl.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        for name, df in sheets:
            _autofit(xl, name, df, header_fmt)
    buf.seek(0)
    return buf.getvalue()


def simulation_to_csv_bytes(sim) -> bytes:
    return sim.table.to_csv(index=False).encode("utf-8")
