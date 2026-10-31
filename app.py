"""
SILK Protein Half-life Calculator -- Streamlit app.

タンパク半減期計算アプリ (SILK: Stable-Isotope-Labelling Kinetics)

Upload the four SILK Excel files (Hsol Pulse / Hsol Pulse&Chase / Hpoi Pulse /
Hpoi Pulse&Chase), choose a kinetic model, and the app fits a turnover half-life
for every protein of interest using precursor-driven or single-exponential
kinetic models.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import os

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from silk_core import pipeline, plotting, export, quality, simulation


def show(fig):
    """Render a matplotlib figure and free it (Streamlit reruns the script)."""
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(page_title="SILK Half-life Calculator", layout="wide", page_icon="🧬")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_FILES = {
    "hsol_pulse": "250714_SILK_Monk_Hsol_Pulse.xlsx",
    "hsol_pandc": "250714_SILK_Monk_Hsol_PandC.xlsx",
    "hpoi_pulse": "250714_SILK_Marmo_Cortex3_Hpoi_Pulse_n1841.xlsx",
    "hpoi_pandc": "250714_SILK_Marmo_Cortex3_Hpoi_PandC_n1841.xlsx",
}


# --------------------------------------------------------------------------- #
#  cached compute
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _run_cached(hsol_pulse_b, hsol_pandc_b, hpoi_pulse_b, hpoi_pandc_b,
                model, fit_mode, tpoint, max_proteins, precursor_source):
    def buf(b):
        return io.BytesIO(b) if b is not None else None

    data = pipeline.load_all(buf(hsol_pulse_b), buf(hsol_pandc_b),
                             buf(hpoi_pulse_b), buf(hpoi_pandc_b))
    result = pipeline.run(
        precursor_pulse=data["precursor_pulse"],
        protein_pulse=data["protein_pulse"],
        precursor_pandc=data["precursor_pandc"],
        protein_pandc=data["protein_pandc"],
        model=model, fit_mode=fit_mode, tpoint=tpoint,
        max_proteins=max_proteins, precursor_source=precursor_source,
    )
    return result


def _read_bytes(uploaded):
    if uploaded is None:
        return None
    return uploaded.getvalue()


def _sample_bytes(key):
    path = os.path.join(SAMPLE_DIR, SAMPLE_FILES[key])
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


# --------------------------------------------------------------------------- #
#  header
# --------------------------------------------------------------------------- #
st.title("🧬 SILK Protein Half-life Calculator")
st.caption("タンパク半減期計算アプリ — pulse / pulse&chase proteomics から各タンパクの半減期を算出")

with st.expander("ℹ️  About this app / このアプリについて", expanded=False):
    st.markdown(
        """
This Python application estimates protein turnover half-lives from **SILK**
(Stable-Isotope-Labelling Kinetics) pulse / pulse-and-chase data.

**Inputs (4 Excel files):**
- **Hsol Pulse** / **Hsol Pulse&Chase** — soluble precursor (free heavy amino-acid)
  Label/Total ratio over time. This is the *forcing function* that drives protein labelling.
- **Hpoi Pulse** / **Hpoi Pulse&Chase** — each protein's Label/Total ratio over time.

**Models:**
- **2-compartment (precursor-driven)** — accounts for the slowly-rising precursor
  (estimated from Hsol measurements or a global protein fit). This is the default
  model and accounts for gradual precursor labelling. Half-life = ln(2) × τ.
- **Single-exponential pool** — assumes instant precursor labelling
  (`LR(t) = 1 − e^(−k·t)`). Half-life = ln(2) / k.

**Fit modes:**
- **Pulse only** — fit each protein to its Pulse time-course (Nature-paper validation path).
- **Pulse + Pulse&Chase** — joint fit using both curves; the chase branch starts at
  the switch time (*tpoint*).
        """
    )

# --------------------------------------------------------------------------- #
#  sidebar -- inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. データファイル / Data files")
    use_sample = st.toggle("Use bundled sample data / サンプルデータを使用", value=False)

    if use_sample:
        hsol_pulse_b = _sample_bytes("hsol_pulse")
        hsol_pandc_b = _sample_bytes("hsol_pandc")
        hpoi_pulse_b = _sample_bytes("hpoi_pulse")
        hpoi_pandc_b = _sample_bytes("hpoi_pandc")
        st.success("Loaded the 4 bundled sample files.")
    else:
        hsol_pulse = st.file_uploader("Hsol Pulse (.xlsx)", type=["xlsx", "xls"])
        hsol_pandc = st.file_uploader("Hsol Pulse&Chase (.xlsx)", type=["xlsx", "xls"])
        hpoi_pulse = st.file_uploader("Hpoi Pulse (.xlsx)", type=["xlsx", "xls"])
        hpoi_pandc = st.file_uploader("Hpoi Pulse&Chase (.xlsx)", type=["xlsx", "xls"])
        hsol_pulse_b = _read_bytes(hsol_pulse)
        hsol_pandc_b = _read_bytes(hsol_pandc)
        hpoi_pulse_b = _read_bytes(hpoi_pulse)
        hpoi_pandc_b = _read_bytes(hpoi_pandc)

    st.header("2. モデル / Model")
    model_label = st.radio(
        "Kinetic model",
        ["2-compartment (precursor-driven)", "Single-exponential pool"],
        index=0,
        help="2-compartment uses a precursor curve from Hsol measurements or a global protein fit.",
    )
    model = "2comp" if model_label.startswith("2-comp") else "single"

    precursor_source = "plasma"
    if model == "2comp":
        prec_label = st.radio(
            "Precursor source / 前駆体の求め方",
            ["Plasma measurement (Hsol files)",
             "Global fit of all proteins (paper method)"],
            index=0,
            help=("Plasma: fit the measured plasma Hsol curve (Takeda-tool behaviour). "
                  "Global: estimate the precursor from all proteins as in Fornasiero "
                  "et al. 2018; plasma is then used only as a validation overlay."),
        )
        precursor_source = "plasma" if prec_label.startswith("Plasma") else "global"

    fit_label = st.radio(
        "Fit mode",
        ["Pulse only", "Pulse + Pulse&Chase"],
        index=0,
    )
    fit_mode = "pulse" if fit_label == "Pulse only" else "pulse+chase"

    st.header("3. オプション / Options")
    auto_tpoint = st.checkbox("Auto-detect chase switch (tpoint)", value=True)
    tpoint = None
    if not auto_tpoint:
        tpoint = st.number_input("Chase switch tpoint (days)", value=14.0, min_value=0.0, step=1.0)

    preview = st.checkbox("Quick preview (first 100 proteins only)", value=False)
    max_proteins = 100 if preview else None

    run_clicked = st.button("▶ Run / 計算実行", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
#  validation of inputs
# --------------------------------------------------------------------------- #
def _requirements_ok():
    problems = []
    if hpoi_pulse_b is None:
        problems.append("Hpoi Pulse file is required.")
    if model == "2comp" and precursor_source == "plasma" and hsol_pulse_b is None:
        problems.append("Plasma precursor mode needs the Hsol Pulse file "
                        "(or switch to the global-fit precursor method).")
    if fit_mode == "pulse+chase" and hpoi_pandc_b is None:
        problems.append("Pulse+Chase fitting needs the Hpoi Pulse&Chase file.")
    return problems


if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked:
    problems = _requirements_ok()
    if problems:
        for p in problems:
            st.error(p)
    else:
        with st.spinner("Fitting precursor and per-protein half-lives… / 計算中…"):
            try:
                st.session_state.result = _run_cached(
                    hsol_pulse_b, hsol_pandc_b, hpoi_pulse_b, hpoi_pandc_b,
                    model, fit_mode, tpoint, max_proteins, precursor_source)
            except Exception as exc:  # surface a clean message
                st.session_state.result = None
                st.error(f"Could not complete the calculation: {exc}")

result = st.session_state.result


# --------------------------------------------------------------------------- #
#  results
# --------------------------------------------------------------------------- #
if result is None:
    st.info("⬅️  Upload the 4 files (or use the sample data) and press **Run**.")
    st.stop()

table = result.table
finite = table["HalfLife_days"].replace([np.inf, -np.inf], np.nan).dropna()

# summary metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Proteins", f"{len(table):,}")
c2.metric("Median half-life", f"{np.median(finite):.1f} d" if len(finite) else "—")
c3.metric("Chase switch", f"{result.tpoint:g} d")
c4.metric("Model", "2-comp" if result.model == "2comp" else "Single-exp")
c5.metric("Median R²", f"{np.nanmedian(table['R2']):.3f}")

tab_over, tab_curve, tab_table, tab_detail, tab_qc, tab_sim = st.tabs(
    ["📈 Overview", "🧬 Lifetime curve (Fig 3i)", "📋 Results table", "🔬 Protein detail",
     "✅ Fit quality (QC)", "🧪 100-day simulation"])

# ---- overview ----
with tab_over:
    left, right = st.columns(2)
    with left:
        prm = result.precursor_params
        is_global = prm.get("method") == "global"
        st.subheader("Precursor — global fit" if is_global else "Precursor (Hsol) — plasma fit")
        if result.model == "2comp":
            show(plotting.precursor_figure(result))
            st.caption(
                f"T1 = {prm['T1']:.2f} d · T2 = {prm['T2']:.2f} d · A = {prm['A']:.3f}"
                + (f" · A1 = {prm['A1']:.3f} · A2 = {prm['A2']:.3f}"
                   if np.isfinite(prm.get("A1", np.nan)) else "")
            )
            if is_global:
                st.caption(
                    f"Estimated from {prm.get('n_proteins_used', 0):,} proteins "
                    "(no plasma used in the fit)."
                )
                r2v = prm.get("r2_pulse", np.nan)
                if np.isfinite(r2v):
                    st.caption(f"Validation vs plasma measurement: R² = {r2v:.3f} "
                               "(paper's Supplementary Fig. 4b check).")
                if prm.get("railed"):
                    st.warning(
                        "⚠️ The global precursor optimum sits on a parameter bound — "
                        "with few time points the precursor is only weakly identified "
                        "from proteins alone (a slow protein rise can be explained by a "
                        "slow precursor **or** slow turnover). The paper stabilised this "
                        "with additional chase / full-labelling data and validated against "
                        "plasma. For this dataset the plasma-measurement mode is more reliable."
                    )
            else:
                st.caption(f"Plasma fit R² = {prm['r2_pulse']:.3f}"
                           + (f" · chase R² = {prm['r2_chase']:.3f}"
                              if np.isfinite(prm.get("r2_chase", np.nan)) else ""))
        else:
            st.info("Single-exponential model does not use a precursor.")
            if result.precursor_pulse is not None:
                show(plotting.precursor_figure(result))
    with right:
        st.subheader("Half-life distribution")
        log_scale = st.checkbox("Log x-axis", value=True)
        show(plotting.halflife_histogram(table, log_scale=log_scale))

    flagged = int((table["Note"] != "").sum())
    if flagged:
        st.caption(f"⚠️ {flagged} proteins carry a quality note "
                   "(at a fit bound or SE not estimable).")

# ---- lifetime curve (Supplementary Fig. 3i style) ----
with tab_curve:
    st.subheader("Label incorporation vs. lifetime")
    st.caption(
        "Model curves give the predicted Label/Total ratio at each pulse time as a "
        "function of protein lifetime; every protein is plotted at (its fitted "
        "half-life, its measured ratio). Adjust the axes below."
    )

    meas_times = result.protein_pulse.time[result.protein_pulse.time > 0]
    _hl = table["HalfLife_days"].to_numpy(dtype=float)
    _hl = _hl[np.isfinite(_hl) & (_hl > 0)]
    x_default = 60.0
    if _hl.size:
        x_default = float(np.clip(np.round(np.nanpercentile(_hl, 95)), 10.0, 400.0))
    _rmax = np.nanmax(result.protein_pulse.ratios) if result.protein_pulse.ratios.size else 0.3

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.1, 1.2])
    with ctrl1:
        st.markdown("**X axis — lifetime (days)**")
        x_min = st.number_input("X min", value=0.0, step=1.0, key="xmin")
        x_max = st.number_input("X max", value=x_default, min_value=0.1, step=5.0, key="xmax")
    with ctrl2:
        st.markdown("**Y axis**")
        in_percent = st.toggle("Show Y as % (else fraction)", value=True, key="ypct")
        y_scale = 100.0 if in_percent else 1.0
        y_top = float(np.round(_rmax * y_scale * 1.2, 1)) or (y_scale)
        y_min = st.number_input("Y min", value=0.0, step=1.0 if in_percent else 0.01, key="ymin")
        y_max = st.number_input("Y max", value=y_top, step=1.0 if in_percent else 0.01, key="ymax")
    with ctrl3:
        st.markdown("**Series**")
        sel_tp = st.multiselect(
            "Pulse time points",
            options=[float(x) for x in meas_times],
            default=[float(x) for x in meas_times],
            format_func=lambda v: f"{v:g} d",
        )
        show_pulse = st.checkbox("Show pulse", value=True)
        show_chase = st.checkbox("Show pulse&chase", value=result.pandc_aligned is not None)
        show_points = st.checkbox("Show protein data points", value=True)

    show(plotting.lifetime_curve_figure(
        result,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        timepoints=sel_tp if sel_tp else None,
        in_percent=in_percent,
        show_pulse=show_pulse,
        show_chase=show_chase,
        show_points=show_points,
    ))
    st.caption("Curves and points use the current model, precursor and fit settings.")


# ---- results table ----
with tab_table:
    st.subheader("Per-protein half-lives")
    query = st.text_input("Filter by gene / protein group / description", "")
    view = table
    if query:
        q = query.lower()
        mask = (
            view["Gene"].astype(str).str.lower().str.contains(q)
            | view["ProteinGroups"].astype(str).str.lower().str.contains(q)
            | view["Description"].astype(str).str.lower().str.contains(q)
        )
        view = view[mask]

    sort_col = st.selectbox("Sort by", ["HalfLife_days", "R2", "Gene", "HalfLife_SE_days"], index=0)
    ascending = st.checkbox("Ascending", value=True)
    view = view.sort_values(sort_col, ascending=ascending, na_position="last")

    st.dataframe(
        view.style.format({
            "HalfLife_days": "{:.2f}", "HalfLife_SE_days": "{:.2f}",
            "CI95_low_days": "{:.2f}", "CI95_high_days": "{:.2f}",
            "FitParameter": "{:.4g}", "R2": "{:.3f}", "SSE": "{:.2e}",
        }, na_rep="—"),
        use_container_width=True, height=460,
    )
    st.caption(f"Showing {len(view):,} of {len(table):,} proteins.")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download Excel (.xlsx)", data=export.results_to_excel_bytes(result),
        file_name="SILK_halflife_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    d2.download_button(
        "⬇️ Download CSV", data=export.results_to_csv_bytes(result),
        file_name="SILK_halflife_results.csv", mime="text/csv",
        use_container_width=True)

# ---- protein detail ----
with tab_detail:
    st.subheader("Inspect a single protein's fit")
    genes = table["Gene"].fillna("").tolist()
    options = [f"{i}: {g or table['ProteinGroups'].iloc[i]}" for i, g in enumerate(genes)]
    sel = st.selectbox("Protein", options, index=0)
    row_index = int(sel.split(":", 1)[0])

    fr = result.fit_results[row_index]
    show(plotting.protein_fit_figure(result, row_index))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Half-life", f"{fr.half_life:.2f} d" if np.isfinite(fr.half_life) else "—")
    m2.metric("± SE", f"{fr.half_life_se:.2f} d" if np.isfinite(fr.half_life_se) else "—")
    m3.metric("95% CI", f"{fr.ci_low:.1f} – {fr.ci_high:.1f} d"
              if np.isfinite(fr.ci_low) else "—")
    m4.metric("R²", f"{fr.r2:.3f}" if np.isfinite(fr.r2) else "—")
    if fr.note:
        st.warning(fr.note)

# ---- fit quality / QC filter (added feature) ----
with tab_qc:
    st.subheader("Per-protein fit-quality filter / fit 品質フィルタ")
    st.caption(
        "Retain a half-life only when its fit meets predefined goodness-of-fit "
        "criteria. Each protein is scored by R², weighted-residual RMSE, the "
        "asymptotic 95 % CI and its size relative to the half-life, plus sampling "
        "completeness. 各タンパクの fit 品質を評価し、基準を満たすものだけを残します。"
    )

    c1, c2, c3 = st.columns(3)
    r2_min = c1.slider("Min R²", 0.0, 1.0, 0.90, 0.01, key="qc_r2")
    rel_ci_max = c2.slider("Max relative 95 % CI", 0.05, 1.0, 0.50, 0.05, key="qc_relci",
                           help="95 % CI expressed relative to the estimated half-life.")
    min_pts = c3.number_input("Min observed points", 1, 20, 3, key="qc_minpts")

    c4, c5, c6 = st.columns(3)
    basis_label = c4.radio("Relative-CI basis", ["± half-width", "full width"],
                           horizontal=True, key="qc_basis",
                           help="± half-width: (CI_high−CI_low)/2 / half-life "
                                "(i.e. the CI as a ± fraction). full width: CI/half-life.")
    max_missing = c5.slider("Max missing fraction", 0.0, 1.0, 0.5, 0.1, key="qc_miss")
    max_hl_in = c6.number_input("Half-life ceiling (days, 0 = none)", 0.0, 5000.0, 0.0, 10.0,
                                key="qc_maxhl")
    exclude_bound = st.checkbox("Exclude estimates pinned to a fit bound", True, key="qc_bound")

    rel_basis = "halfwidth" if basis_label.startswith("±") else "width"
    th = quality.GOFThresholds(
        r2_min=r2_min, rel_ci_max=rel_ci_max, rel_ci_basis=rel_basis,
        min_observed=int(min_pts), max_missing_frac=max_missing,
        exclude_at_bound=exclude_bound, require_finite_ci=True,
        max_halflife_days=None if max_hl_in <= 0 else float(max_hl_in))

    qdf = quality.compute_quality_table(result)
    qdf = quality.apply_gof_filter(qdf, th)
    summ = quality.qc_summary(qdf)

    sim_cutoff = st.session_state.get("sim_cutoff")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Retained (QC pass)", f"{summ['n_pass']:,}")
    q2.metric("Filtered out", f"{summ['n_fail']:,}")
    q3.metric("Retained fraction",
              f"{summ['pass_frac']*100:.1f}%" if np.isfinite(summ['pass_frac']) else "—")
    q4.metric("Median HL (retained)",
              f"{summ['median_hl_pass']:.1f} d" if np.isfinite(summ['median_hl_pass']) else "—")

    st.caption(f"**Criteria:** {th.describe()}")
    show(plotting.quality_scatter_figure(qdf, rel_ci_max=rel_ci_max,
                                         rel_ci_basis=rel_basis, cutoff_days=sim_cutoff))

    st.markdown("**Per-time-point missingness (input data)**")
    st.dataframe(quality.per_timepoint_missing(result.protein_pulse)
                 .style.format({"MissingFrac": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

    pass_only = st.checkbox("Show retained proteins only", True, key="qc_passonly")
    view_q = qdf[qdf["QC_pass"]] if pass_only else qdf
    show_cols = ["ProteinGroups", "Gene", "HalfLife_days", "CI95_low_days", "CI95_high_days",
                 "RelCI_halfwidth", "RelCI_width", "CV", "R2", "RMSE",
                 "N_observed", "MissingFrac", "N_peptides", "N_replicates",
                 "AtBound", "QC_pass", "QC_reason"]
    show_cols = [c for c in show_cols if c in view_q.columns]
    st.dataframe(
        view_q[show_cols].style.format({
            "HalfLife_days": "{:.2f}", "CI95_low_days": "{:.2f}", "CI95_high_days": "{:.2f}",
            "RelCI_halfwidth": "{:.3f}", "RelCI_width": "{:.3f}", "CV": "{:.3f}",
            "R2": "{:.3f}", "RMSE": "{:.4f}", "MissingFrac": "{:.1%}",
        }, na_rep="—"),
        use_container_width=True, height=380)
    st.caption(f"Showing {len(view_q):,} of {len(qdf):,} proteins.")

    dq1, dq2 = st.columns(2)
    dq1.download_button(
        "⬇️ Download QC table (Excel)",
        data=export.quality_to_excel_bytes(result, view_q, th,
                                           missing_df=quality.per_timepoint_missing(result.protein_pulse)),
        file_name="SILK_fit_quality_QC.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
    dq2.download_button(
        "⬇️ Download QC table (CSV)", data=export.quality_to_csv_bytes(view_q),
        file_name="SILK_fit_quality_QC.csv", mime="text/csv", use_container_width=True)

    with st.expander("🔁 Add residual-bootstrap 95 % CIs (non-asymptotic, slower)"):
        st.caption("Resamples each protein's residuals to obtain a percentile CI — a check "
                   "on the asymptotic CI that avoids negative lower bounds for noisy fits.")
        bcol1, bcol2, bcol3 = st.columns(3)
        n_boot = bcol1.number_input("Bootstrap replicates", 100, 2000, 400, 100, key="boot_n")
        cap = bcol2.number_input("Max proteins (retained subset)", 20, 4000, 500, 20, key="boot_cap")
        boot_seed = bcol3.number_input("Seed", 0, 9999, 0, key="boot_seed")
        if st.button("Run bootstrap on retained proteins", key="boot_run"):
            idx = qdf.index[qdf["QC_pass"]].tolist()[: int(cap)]
            row_idx = qdf.loc[idx, "index"].tolist()
            with st.spinner(f"Bootstrapping {len(row_idx)} proteins × {int(n_boot)}…"):
                bdf = quality.bootstrap_ci_batch(result, indices=row_idx,
                                                 n_boot=int(n_boot), seed=int(boot_seed))
            st.session_state["boot_df"] = bdf
        bdf = st.session_state.get("boot_df")
        if bdf is not None and len(bdf):
            merged = bdf.merge(
                qdf[["index", "HalfLife_days", "CI95_low_days", "CI95_high_days"]],
                on="index", how="left")
            st.dataframe(
                merged[["Gene", "HalfLife_days", "CI95_low_days", "CI95_high_days",
                        "Boot_CI95_low_days", "Boot_CI95_high_days", "Boot_median_days",
                        "Boot_SE_days"]].style.format({
                    "HalfLife_days": "{:.2f}", "CI95_low_days": "{:.2f}", "CI95_high_days": "{:.2f}",
                    "Boot_CI95_low_days": "{:.2f}", "Boot_CI95_high_days": "{:.2f}",
                    "Boot_median_days": "{:.2f}", "Boot_SE_days": "{:.2f}"}, na_rep="—"),
                use_container_width=True, height=300)

    st.info(
        "**Suggested methods sentence.** Protein half-life estimates were retained only "
        "when the model fit met predefined goodness-of-fit criteria "
        f"(R² ≥ {r2_min:g}) and the 95 % confidence interval was below "
        f"{rel_ci_max*100:.0f} % of the estimated half-life.")


# ---- 100-day recovery simulation (added feature) ----
with tab_sim:
    st.subheader("Half-life recovery simulation / 100 日相当のシミュレーション")
    st.caption(
        "Justifies the reliable half-life range with the study's own measurement noise. "
        "Using the fitted precursor curve and the model as configured, known half-lives are "
        "sampled at the real time points, given the measured per-time-point noise and "
        "missingness, then re-estimated. 真値を既知として実測ノイズ・欠測を付加し再推定、"
        "bias・95 % CI 被覆・相対誤差を評価します。"
    )

    s1, s2, s3 = st.columns(3)
    hl_text = s1.text_input("True half-lives (days, comma-separated)",
                            "0.5, 1, 5, 20, 50, 75, 100, 125, 150, 200", key="sim_hls")
    n_rep = s2.number_input("Replicates per half-life", 50, 5000, 2000, 50, key="sim_nrep")
    seed = s3.number_input("Random seed", 0, 9999, 123, key="sim_seed")

    s4, s5, s6 = st.columns(3)
    noise_scale = s4.slider("Noise scale (× measured)", 0.25, 3.0, 1.0, 0.25, key="sim_noise")
    extra_missing = s5.slider("Extra missing per point", 0.0, 0.8, 0.0, 0.05, key="sim_miss")
    robust = s6.checkbox("Robust noise (MAD)", True, key="sim_robust")

    n1, n2, n3 = st.columns([2, 1, 1])
    noise_sel_label = n1.selectbox(
        "Noise estimated from / ノイズ推定に使うタンパク",
        list(simulation.NOISE_SELECTIONS.values()), index=0, key="sim_noise_sel",
        help="Default: every protein whose fitted half-life lies within the analysis "
             "range (no R² filter) — i.e. the proteins carried into the downstream analysis.")
    noise_sel = [k for k, v in simulation.NOISE_SELECTIONS.items() if v == noise_sel_label][0]
    hl_lo = n2.number_input("Analysis range min (days)", 0.0, 1000.0, 1.0, 0.5, key="sim_rng_lo",
                            disabled=noise_sel != "analysis_range")
    hl_hi = n3.number_input("Analysis range max (days)", 0.1, 5000.0, 100.0, 5.0, key="sim_rng_hi",
                            disabled=noise_sel != "analysis_range")

    with st.expander("Acceptance criteria for the reliable-recovery range"):
        a1, a2, a3 = st.columns(3)
        pct_bias_tol = a1.slider("Max |bias| (%)", 5.0, 50.0, 15.0, 5.0, key="sim_bias")
        coverage_min = a2.slider("Min 95 % CI coverage", 0.5, 0.99, 0.90, 0.01, key="sim_cov")
        rel_error_tol = a3.slider("Max median relative error", 0.1, 1.0, 0.50, 0.05, key="sim_re")

    if result.model == "2comp" and not np.isfinite(result.precursor_params.get("T1", np.nan)):
        st.warning("The 2-compartment precursor is not available — run the pipeline first.")
    run_sim = st.button("▶ Run simulation / シミュレーション実行", type="primary", key="sim_run")

    if run_sim:
        try:
            true_hls = [float(x) for x in hl_text.replace("，", ",").split(",") if x.strip()]
        except ValueError:
            true_hls = list(simulation.DEFAULT_TRUE_HALFLIVES)
            st.warning("Could not parse the half-life list — using the default set.")
        with st.spinner("Simulating recovery under measured noise… / シミュレーション中…"):
            sim = simulation.simulate_recovery(
                result, true_halflives=true_hls, n_rep=int(n_rep),
                noise_scale=noise_scale, extra_missing=extra_missing,
                seed=int(seed), robust=robust,
                noise_selection=noise_sel, hl_range=(float(hl_lo), float(hl_hi)))
            rec = simulation.recommend_cutoff(
                sim, pct_bias_tol=pct_bias_tol, coverage_min=coverage_min,
                rel_error_tol=rel_error_tol)
        st.session_state["sim"] = sim
        st.session_state["sim_rec"] = rec
        st.session_state["sim_cutoff"] = rec.get("cutoff_days")

    sim = st.session_state.get("sim")
    rec = st.session_state.get("sim_rec")
    if sim is None:
        st.info("Set the true half-lives and press **Run simulation**. "
                "真値を設定して実行してください。")
    else:
        if rec.get("cutoff_days"):
            st.success(f"**{rec['headline_en']}**\n\n{rec['headline_jp']}")
        else:
            st.warning(f"**{rec['headline_en']}**\n\n{rec['headline_jp']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Reliable range",
                  f"{rec['lower_days']:g}–{rec['cutoff_days']:g} d" if rec.get("cutoff_days") else "—")
        m2.metric("Model", "2-comp" if sim.model == "2comp" else "Single-exp")
        m3.metric("Noise SD (pooled)", f"{sim.noise.pooled_sd:.4f}",
                  help=f"Estimated from {sim.noise.n_used:,} of {sim.noise.n_total:,} proteins")

        show_header = not st.checkbox(
            "Publication style — hide the figure header / 論文用（図の見出しを表示しない）",
            value=False, key="sim_pub")
        sim_fig = plotting.simulation_figure(
            sim, cutoff_days=rec.get("cutoff_days"),
            pct_bias_tol=rec["criteria"]["pct_bias_tol"],
            coverage_min=rec["criteria"]["coverage_min"],
            rel_error_tol=rec["criteria"]["rel_error_tol"],
            show_header=show_header)
        f1, f2, f3 = st.columns(3)
        f1.download_button("⬇️ Figure PNG (300 dpi)", plotting.figure_to_bytes(sim_fig, "png", 300),
                           file_name="SILK_recovery_simulation.png", mime="image/png",
                           use_container_width=True)
        f2.download_button("⬇️ Figure PDF (vector)", plotting.figure_to_bytes(sim_fig, "pdf"),
                           file_name="SILK_recovery_simulation.pdf", mime="application/pdf",
                           use_container_width=True)
        f3.download_button("⬇️ Figure SVG (vector)", plotting.figure_to_bytes(sim_fig, "svg"),
                           file_name="SILK_recovery_simulation.svg", mime="image/svg+xml",
                           use_container_width=True)
        show(sim_fig)

        st.markdown("**Per-true-half-life recovery metrics**")
        st.dataframe(
            sim.table.style.format({
                "TrueHalfLife_days": "{:.0f}", "Median_est_days": "{:.1f}",
                "Mean_est_days": "{:.1f}", "Bias_days": "{:.2f}", "PctBias": "{:.1f}",
                "RelError_median": "{:.3f}", "Coverage95": "{:.3f}",
                "CI95_width_median_days": "{:.1f}", "RelCI_median": "{:.2f}",
                "FracFinite": "{:.2f}", "FracAtBound": "{:.2f}"}, na_rep="—"),
            use_container_width=True, hide_index=True)
        st.caption(f"Noise model — {sim.noise.describe()}")

        ds1, ds2 = st.columns(2)
        ds1.download_button(
            "⬇️ Download simulation (Excel)",
            data=export.simulation_to_excel_bytes(sim, rec),
            file_name="SILK_recovery_simulation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
        ds2.download_button(
            "⬇️ Download simulation (CSV)", data=export.simulation_to_csv_bytes(sim),
            file_name="SILK_recovery_simulation.csv", mime="text/csv",
            use_container_width=True)

        st.info(f"**Suggested manuscript sentence.** {rec['manuscript_en']}")


st.divider()
st.caption("SILK Half-life Calculator · Protein turnover analysis with Python and Streamlit.")
