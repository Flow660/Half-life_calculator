"""
Headless validation of the SILK half-life core on the four provided files.

Runs the whole pipeline (precursor fit + per-protein half-life fit) for both
models and both fit modes, and prints diagnostics.  Uses numpy only.

Usage:  python validate.py /path/to/input_folder
"""

import sys
import time

import numpy as np

from silk_core import loaders, precursor as prec_mod, pipeline

INPUT = sys.argv[1] if len(sys.argv) > 1 else "../../input"

F = {
    "hsol_pulse": f"{INPUT}/250714_SILK_Monk_Hsol_Pulse.xlsx",
    "hsol_pandc": f"{INPUT}/250714_SILK_Monk_Hsol_PandC.xlsx",
    "hpoi_pulse": f"{INPUT}/250714_SILK_Marmo_Cortex3_Hpoi_Pulse_n1841.xlsx",
    "hpoi_pandc": f"{INPUT}/250714_SILK_Marmo_Cortex3_Hpoi_PandC_n1841.xlsx",
}


def hr(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def main():
    hr("1. LOAD")
    pp = loaders.load_precursor(F["hsol_pulse"])
    pc = loaders.load_precursor(F["hsol_pandc"])
    hp = loaders.load_protein(F["hpoi_pulse"])
    hc = loaders.load_protein(F["hpoi_pandc"])
    print(f"Hsol Pulse : time={pp.time.tolist()}")
    print(f"             ratio={np.round(pp.ratio, 4).tolist()}")
    print(f"Hsol P&C   : time={pc.time.tolist()}")
    print(f"             ratio={np.round(pc.ratio, 4).tolist()}")
    print(f"Hpoi Pulse : {hp.n_proteins} proteins, time={hp.time.tolist()}")
    print(f"Hpoi P&C   : {hc.n_proteins} proteins, time={hc.time.tolist()}")
    print(f"First proteins: {hp.genes[:6]}")

    hr("2. PRECURSOR FIT (Hsol bi-exponential)")
    tpoint = loaders.infer_chase_timepoint(hp, hc)
    print(f"Inferred Pulse->Chase switch (tpoint) = {tpoint} days")
    prm = prec_mod.fit_precursor(pp, pc, tpoint=tpoint)
    print(f"T1={prm['T1']:.4f}  T2={prm['T2']:.4f}  A={prm['A']:.4f}")
    print(f"A1={prm['A1']:.4f}  A2={prm['A2']:.4f}  (chase amplitudes)")
    print(f"R2 pulse={prm['r2_pulse']:.4f}  R2 chase={prm['r2_chase']:.4f}")

    for model in ("single", "2comp"):
        for fit_mode in ("pulse", "pulse+chase"):
            hr(f"3. HALF-LIFE FIT  model={model}  fit_mode={fit_mode}  (first 40 proteins)")
            t0 = time.time()
            r = pipeline.run(
                precursor_pulse=pp, protein_pulse=hp,
                precursor_pandc=pc, protein_pandc=hc,
                model=model, fit_mode=fit_mode, tpoint=tpoint, max_proteins=40)
            dt = time.time() - t0
            df = r.table
            good = df[np.isfinite(df["HalfLife_days"])]
            print(f"fitted {len(df)} proteins in {dt:.2f}s")
            print(df[["Gene", "HalfLife_days", "HalfLife_SE_days",
                      "CI95_low_days", "CI95_high_days", "R2"]].head(8).to_string(index=False))
            hl = good["HalfLife_days"].to_numpy()
            hl = hl[np.isfinite(hl)]
            if hl.size:
                print(f"half-life days -> min={hl.min():.2f}  median={np.median(hl):.2f}"
                      f"  mean={hl.mean():.2f}  max={hl.max():.2f}")

    hr("4. FULL RUN TIMING (all proteins, single/pulse)")
    t0 = time.time()
    r = pipeline.run(precursor_pulse=pp, protein_pulse=hp,
                     precursor_pandc=pc, protein_pandc=hc,
                     model="single", fit_mode="pulse", tpoint=tpoint)
    dt = time.time() - t0
    df = r.table
    finite = np.isfinite(df["HalfLife_days"]).sum()
    print(f"all {len(df)} proteins in {dt:.2f}s  ({finite} finite half-lives)")
    print("Validation complete.")


if __name__ == "__main__":
    main()
