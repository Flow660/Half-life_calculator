"""
End-to-end SILK half-life pipeline.

    load 4 files  ->  fit precursor (Hsol)  ->  fit every protein (Hpoi)  ->  table

The public entry point is :func:`run`, which accepts already-loaded data objects
(see :mod:`silk_core.loaders`) so it can be driven equally well from the Streamlit
UI or from a headless script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import loaders, precursor as precursor_mod, fitting


MODELS = {
    "single": "Single-exponential pool model",
    "2comp": "2-compartment (precursor-driven) model",
}
FIT_MODES = {
    "pulse": "Pulse only (Nature-paper validation)",
    "pulse+chase": "Pulse + Pulse&Chase (joint fit)",
}


@dataclass
class RunResult:
    table: pd.DataFrame
    precursor_params: dict
    tpoint: float
    model: str
    fit_mode: str
    protein_pulse: loaders.ProteinData
    protein_pandc: loaders.ProteinData | None
    precursor_pulse: loaders.PrecursorData
    precursor_pandc: loaders.PrecursorData | None
    fit_results: list          # list[fitting.FitResult]
    pandc_aligned: object = None   # (N, T) Pulse&Chase ratios aligned to the Pulse rows
    precursor_source: str = "plasma"


def load_all(hsol_pulse, hsol_pandc, hpoi_pulse, hpoi_pandc):
    """Load the four SILK files into dataclass objects."""
    return {
        "precursor_pulse": loaders.load_precursor(hsol_pulse) if hsol_pulse is not None else None,
        "precursor_pandc": loaders.load_precursor(hsol_pandc) if hsol_pandc is not None else None,
        "protein_pulse": loaders.load_protein(hpoi_pulse) if hpoi_pulse is not None else None,
        "protein_pandc": loaders.load_protein(hpoi_pandc) if hpoi_pandc is not None else None,
    }


def run(
    precursor_pulse: loaders.PrecursorData,
    protein_pulse: loaders.ProteinData,
    precursor_pandc: loaders.PrecursorData | None = None,
    protein_pandc: loaders.ProteinData | None = None,
    model: str = "2comp",
    fit_mode: str = "pulse",
    tpoint: float | None = None,
    max_proteins: int | None = None,
    precursor_source: str = "plasma",
    progress=None,
) -> RunResult:
    """
    Run the pipeline.

    Parameters
    ----------
    precursor_pulse / precursor_pandc : Hsol (plasma) data. Required for the
        2-compartment model when precursor_source='plasma'; optional (validation
        overlay only) when precursor_source='global'.
    protein_pulse   / protein_pandc   : Hpoi data (pandc required for joint fit).
    model     : 'single' or '2comp'.
    fit_mode  : 'pulse' or 'pulse+chase'.
    tpoint    : Pulse->Chase switch (days). Inferred if None.
    max_proteins : cap for previews (None = all).
    precursor_source : '2comp' precursor origin --
        'plasma' : fit the measured plasma Hsol curve (Takeda-tool behaviour).
        'global' : estimate the precursor by globally fitting all proteins
                   (Fornasiero et al. 2018 method); plasma is used only as a
                   validation overlay if provided.
    progress  : optional callable(done:int, total:int).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    if fit_mode not in FIT_MODES:
        raise ValueError(f"fit_mode must be one of {list(FIT_MODES)}")
    if precursor_source not in ("plasma", "global"):
        raise ValueError("precursor_source must be 'plasma' or 'global'")

    fit_pandc = fit_mode == "pulse+chase"
    if fit_pandc and protein_pandc is None:
        raise ValueError("Pulse+Chase fitting needs the Hpoi Pulse&Chase file.")

    # ---- chase switch time ----
    if tpoint is None:
        if protein_pandc is not None:
            tpoint = loaders.infer_chase_timepoint(protein_pulse, protein_pandc)
        else:
            t = protein_pulse.time
            tpoint = float(t[len(t) // 2]) if t.size else 14.0

    # ---- align protein rows between pulse and pandc by protein-group id ----
    t = protein_pulse.time
    pulse_ratios_full = protein_pulse.ratios
    n_proteins = protein_pulse.n_proteins
    if max_proteins is not None:
        n_proteins = min(n_proteins, max_proteins)
    pulse_ratios = pulse_ratios_full[:n_proteins]

    # ---- precursor for the 2-compartment model ----
    if model == "2comp":
        if precursor_source == "plasma":
            if precursor_pulse is None:
                raise ValueError("Plasma precursor mode needs the Hsol Pulse file.")
            prec_params = precursor_mod.fit_precursor(
                precursor_pulse, precursor_pandc,
                tpoint=tpoint if precursor_pandc else None)
            prec_params["method"] = "plasma"
        else:  # global (paper method) -- estimate from all proteins
            prec_params = precursor_mod.estimate_precursor_global(
                t, protein_pulse.ratios,
                protein_pandc.ratios if protein_pandc is not None else None,
                tpoint=tpoint, fit_pandc=fit_pandc)
            prec_params["method"] = "global"
            # validation overlay: compare estimated curve with plasma, if provided
            if precursor_pulse is not None:
                pt = np.asarray(precursor_pulse.time, dtype=float)
                py = np.asarray(precursor_pulse.ratio, dtype=float)
                prec_params["r2_pulse"] = precursor_mod._r2(
                    py, precursor_mod.precursor_curve(prec_params, pt, chase=False))
                prec_params["pulse_data"] = (pt, py)
    else:
        prec_params = {"T1": np.nan, "T2": np.nan, "A": np.nan,
                       "A1": np.nan, "A2": np.nan, "tpoint": tpoint,
                       "r2_pulse": np.nan, "r2_chase": np.nan,
                       "method": "n/a", "pulse_data": None, "chase_data": None}

    labels = []
    for i in range(n_proteins):
        labels.append((
            protein_pulse.protein_groups[i] if i < len(protein_pulse.protein_groups) else "",
            protein_pulse.genes[i] if i < len(protein_pulse.genes) else "",
            protein_pulse.descriptions[i] if i < len(protein_pulse.descriptions) else "",
        ))

    # build the aligned Pulse&Chase matrix (matched by protein-group id)
    pandc_ratios = None
    if protein_pandc is not None:
        pandc_lookup = {}
        for j, pg in enumerate(protein_pandc.protein_groups):
            pandc_lookup.setdefault(pg, j)
        pandc_ratios = np.full((n_proteins, t.size), np.nan)
        for i in range(n_proteins):
            j = pandc_lookup.get(protein_pulse.protein_groups[i])
            if j is None and i < protein_pandc.n_proteins:
                j = i  # positional fallback
            if j is not None:
                pandc_ratios[i] = protein_pandc.ratios[j]

    if progress is not None:
        progress(0, n_proteins)
    results = fitting.fit_batch(
        t, pulse_ratios, pandc_ratios, tpoint,
        model=model, precursor=prec_params, fit_pandc=fit_pandc,
        labels=labels, progress=progress)
    if progress is not None:
        progress(n_proteins, n_proteins)

    table = _build_table(results)
    prec_params = dict(prec_params)
    prec_params["tpoint"] = tpoint

    return RunResult(
        table=table,
        precursor_params=prec_params,
        tpoint=float(tpoint),
        model=model,
        fit_mode=fit_mode,
        protein_pulse=protein_pulse,
        protein_pandc=protein_pandc,
        precursor_pulse=precursor_pulse,
        precursor_pandc=precursor_pandc,
        fit_results=results,
        pandc_aligned=pandc_ratios,
        precursor_source=precursor_source,
    )


def _build_table(results) -> pd.DataFrame:
    rows = [r.as_row() for r in results]
    df = pd.DataFrame(rows)
    rename = {
        "protein_group": "ProteinGroups",
        "gene": "Gene",
        "description": "Description",
        "half_life": "HalfLife_days",
        "half_life_se": "HalfLife_SE_days",
        "ci_low": "CI95_low_days",
        "ci_high": "CI95_high_days",
        "parameter": "FitParameter",
        "r2": "R2",
        "sse": "SSE",
        "n_points": "N_points",
        "model": "Model",
        "fit_mode": "FitMode",
        "note": "Note",
    }
    df = df.rename(columns=rename)
    ordered = ["index", "ProteinGroups", "Gene", "Description",
               "HalfLife_days", "HalfLife_SE_days", "CI95_low_days", "CI95_high_days",
               "FitParameter", "R2", "SSE", "N_points", "Model", "FitMode", "Note"]
    df = df[[c for c in ordered if c in df.columns]]
    return df
