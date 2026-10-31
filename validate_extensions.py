"""
Headless numeric validation of the two added features on the four SILK files.

Run:  python validate_extensions.py /path/to/input_folder

Exercises, with no Streamlit:
  1. the per-protein fit-quality filter (R2 / relative CI / missingness / bootstrap CI)
  2. the noise-driven half-life recovery simulation + data-driven cutoff
and confirms the figures render and the Excel/CSV exports write.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")

from silk_core import loaders, pipeline, quality, simulation, plotting, export

INP = sys.argv[1] if len(sys.argv) > 1 else "../../input"
F = {
    "hsol_pulse": f"{INP}/250714_SILK_Monk_Hsol_Pulse.xlsx",
    "hsol_pandc": f"{INP}/250714_SILK_Monk_Hsol_PandC.xlsx",
    "hpoi_pulse": f"{INP}/250714_SILK_Marmo_Cortex3_Hpoi_Pulse_n1841.xlsx",
    "hpoi_pandc": f"{INP}/250714_SILK_Marmo_Cortex3_Hpoi_PandC_n1841.xlsx",
}

def hr(m): print("\n" + "=" * 78 + f"\n{m}\n" + "=" * 78)

def main():
    hr("LOAD + PIPELINE (2-compartment, pulse) — unchanged core")
    pp = loaders.load_precursor(F["hsol_pulse"]); pc = loaders.load_precursor(F["hsol_pandc"])
    hp = loaders.load_protein(F["hpoi_pulse"]);   hc = loaders.load_protein(F["hpoi_pandc"])
    res = pipeline.run(precursor_pulse=pp, protein_pulse=hp, precursor_pandc=pc,
                       protein_pandc=hc, model="2comp", fit_mode="pulse")
    print(f"{len(res.table)} proteins; sampling times = {hp.time[hp.time>0].tolist()} days")

    hr("FEATURE 1 — per-protein fit-quality filter")
    q = quality.compute_quality_table(res)
    th = quality.GOFThresholds(r2_min=0.90, rel_ci_max=0.30, rel_ci_basis="halfwidth")
    q = quality.apply_gof_filter(q, th)
    s = quality.qc_summary(q)
    print("criteria:", th.describe())
    print(f"retained {s['n_pass']}/{s['n_total']} ({s['pass_frac']*100:.1f}%); "
          f"median HL retained = {s['median_hl_pass']:.1f} d")
    print("columns:", ", ".join(q.columns))
    mt = quality.per_timepoint_missing(hp)
    print("missingness:", mt.to_dict("records"))
    t0 = time.time()
    b = quality.bootstrap_ci_batch(res, indices=q.index[q['QC_pass']].tolist()[:60], n_boot=200, seed=0)
    print(f"bootstrap 60×200 in {time.time()-t0:.2f}s; e.g. {b.iloc[0].to_dict()}")

    hr("FEATURE 2 — recovery simulation + cutoff")
    sim = simulation.simulate_recovery(res, n_rep=400, seed=1)
    print(sim.table[["TrueHalfLife_days","Median_est_days","PctBias","RelError_median",
                     "Coverage95","RelCI_median","FracAtBound"]].to_string(index=False))
    rec = simulation.recommend_cutoff(sim)
    print("\ncutoff:", rec["cutoff_days"], "| all_pass:", rec["all_pass"], "| gradual:", rec["gradual_degradation"])
    print("EN:", rec["manuscript_en"])
    print("JP:", rec["headline_jp"])

    hr("FIGURES + EXPORTS render/write")
    outdir = "_qa_out"; os.makedirs(outdir, exist_ok=True)
    fig1 = plotting.quality_scatter_figure(q, rel_ci_max=0.30, cutoff_days=rec["cutoff_days"])
    fig1.savefig(f"{outdir}/qc_scatter.png", dpi=90); print("qc_scatter.png OK")
    fig2 = plotting.simulation_figure(sim, cutoff_days=rec["cutoff_days"])
    fig2.savefig(f"{outdir}/simulation.png", dpi=90); print("simulation.png OK")
    with open(f"{outdir}/qc.xlsx","wb") as fh: fh.write(export.quality_to_excel_bytes(res, q[q['QC_pass']], th, mt))
    with open(f"{outdir}/sim.xlsx","wb") as fh: fh.write(export.simulation_to_excel_bytes(sim, rec))
    print("QC + simulation Excel written:",
          os.path.getsize(f"{outdir}/qc.xlsx"), "/", os.path.getsize(f"{outdir}/sim.xlsx"), "bytes")
    print("\nAll extension checks passed.")

if __name__ == "__main__":
    main()
