"""
Readers for the four SILK Excel input formats.

The layouts mirror the MATLAB toolbox (functions F931 / F933 / F934):

Hsol (soluble precursor) file  -- e.g. "..._Hsol_Pulse.xlsx"
    Row 1               : header row. Data columns (from column D onward) hold
                          the time-point labels (Days).
    Row 2               : the precursor Label/Total ratio at each time point.
    Columns A / B / C   : free-text labels (source file, gene name, ...), ignored.
    A t = 0 point with ratio 0 is prepended (matches F933).

Hpoi (protein of interest) file -- e.g. "..._Hpoi_Pulse_n1841.xlsx"
    Row 1               : header row. Columns A/B/C = ProteinGroups /
                          ProteinDescriptions / Gene Name. Data columns
                          (from column D) hold time-point labels (Days).
    Rows 2..N           : one protein per row, its Label/Total ratio per time point.
    A t = 0 point with ratio 0 is prepended for every protein (matches F934).

The data column start (column D = index 3) is the ``DATA_START_COL`` default,
exactly as in the MATLAB code (``nac01 = 4`` in 1-based indexing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# MATLAB nac01 = 4 (1-based, column D). 0-based -> 3.
DATA_START_COL = 3


# --------------------------------------------------------------------------- #
#  Data containers
# --------------------------------------------------------------------------- #
@dataclass
class PrecursorData:
    """Soluble precursor (Hsol) time course."""

    time: np.ndarray          # days, includes the prepended 0
    ratio: np.ndarray         # Label/Total ratio, includes the prepended 0
    source: str = ""          # free-text label from column A
    sheet: str = ""

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.ratio = np.asarray(self.ratio, dtype=float)


@dataclass
class ProteinData:
    """One row per protein of a Hpoi time course."""

    time: np.ndarray                 # days, shape (T,), includes prepended 0
    ratios: np.ndarray               # shape (N_proteins, T), includes prepended 0 col
    protein_groups: list = field(default_factory=list)
    descriptions: list = field(default_factory=list)
    genes: list = field(default_factory=list)
    sheet: str = ""

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.ratios = np.asarray(self.ratios, dtype=float)

    @property
    def n_proteins(self) -> int:
        return self.ratios.shape[0]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _seek0(path_or_buffer):
    """Rewind a file-like object so it can be read more than once."""
    seek = getattr(path_or_buffer, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except Exception:
            pass


def _read_sheet(path_or_buffer, sheet_name=0) -> pd.DataFrame:
    """Read an Excel sheet with no header interpretation (raw grid)."""
    _seek0(path_or_buffer)
    return pd.read_excel(path_or_buffer, sheet_name=sheet_name, header=None, engine="openpyxl")


def _numeric(series) -> np.ndarray:
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)


def _time_labels(header_row, data_start_col=DATA_START_COL) -> np.ndarray:
    """Extract numeric time-point labels from the header row, from data_start_col on."""
    labels = _numeric(header_row[data_start_col:])
    return labels


# --------------------------------------------------------------------------- #
#  Public readers
# --------------------------------------------------------------------------- #
def load_precursor(path_or_buffer, sheet_name=0, data_start_col=DATA_START_COL) -> PrecursorData:
    """
    Read a Hsol (soluble precursor) Pulse or Pulse&Chase file.

    Returns a :class:`PrecursorData` with a prepended (t=0, ratio=0) sample.
    """
    df = _read_sheet(path_or_buffer, sheet_name)
    if df.shape[0] < 2:
        raise ValueError("Precursor file must have a header row and at least one data row.")

    header = df.iloc[0].tolist()
    times = _time_labels(header, data_start_col)

    # The precursor row is the first row (after the header) that has numeric data.
    ratio = None
    for r in range(1, df.shape[0]):
        candidate = _numeric(df.iloc[r].tolist()[data_start_col:])
        if np.isfinite(candidate).any():
            ratio = candidate
            break
    if ratio is None:
        raise ValueError("No numeric precursor data row found.")

    # Keep only columns where a time label exists.
    keep = np.isfinite(times)
    times = times[keep]
    ratio = ratio[keep]

    # Sort by time, then prepend t = 0.
    order = np.argsort(times)
    times, ratio = times[order], ratio[order]
    times = np.concatenate([[0.0], times])
    ratio = np.concatenate([[0.0], ratio])

    src = str(df.iloc[0, 0]) if pd.notna(df.iloc[0, 0]) else ""
    sheet = _resolve_sheet_name(path_or_buffer, sheet_name)
    return PrecursorData(time=times, ratio=ratio, source=src, sheet=sheet)


def load_protein(path_or_buffer, sheet_name=0, data_start_col=DATA_START_COL) -> ProteinData:
    """
    Read a Hpoi (protein of interest) Pulse or Pulse&Chase file.

    Returns a :class:`ProteinData`; every protein gets a prepended (t=0, ratio=0)
    column, matching the MATLAB F934 behaviour.
    """
    df = _read_sheet(path_or_buffer, sheet_name)
    if df.shape[0] < 2:
        raise ValueError("Protein file must have a header row and at least one protein row.")

    header = df.iloc[0].tolist()
    times = _time_labels(header, data_start_col)
    keep = np.isfinite(times)
    times = times[keep]
    order = np.argsort(times)
    times = times[order]

    body = df.iloc[1:].reset_index(drop=True)

    # Label columns A/B/C (best-effort; some files only fill a subset).
    def _col(idx):
        if idx < body.shape[1]:
            return body.iloc[:, idx].astype("object").where(body.iloc[:, idx].notna(), "").tolist()
        return [""] * body.shape[0]

    protein_groups = [str(x) for x in _col(0)]
    descriptions = [str(x) for x in _col(1)]
    genes = [str(x) for x in _col(2)]

    # Numeric ratio block.
    raw = body.iloc[:, data_start_col:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw = raw[:, keep][:, order]  # keep valid time cols, sort by time

    # Prepend the t = 0 (ratio = 0) column for every protein.
    zeros = np.zeros((raw.shape[0], 1))
    ratios = np.hstack([zeros, raw])
    times = np.concatenate([[0.0], times])

    # Drop rows that are entirely empty (no finite measurement).
    finite_rows = np.isfinite(raw).any(axis=1)
    ratios = ratios[finite_rows]
    protein_groups = [g for g, k in zip(protein_groups, finite_rows) if k]
    descriptions = [d for d, k in zip(descriptions, finite_rows) if k]
    genes = [g for g, k in zip(genes, finite_rows) if k]

    sheet = _resolve_sheet_name(path_or_buffer, sheet_name)
    return ProteinData(
        time=times,
        ratios=ratios,
        protein_groups=protein_groups,
        descriptions=descriptions,
        genes=genes,
        sheet=sheet,
    )


def _resolve_sheet_name(path_or_buffer, sheet_name) -> str:
    try:
        _seek0(path_or_buffer)
        xls = pd.ExcelFile(path_or_buffer, engine="openpyxl")
        if isinstance(sheet_name, int):
            return xls.sheet_names[sheet_name]
        return str(sheet_name)
    except Exception:
        return str(sheet_name)


def infer_chase_timepoint(protein_pulse: ProteinData, protein_pandc: ProteinData) -> float:
    """
    Infer the Pulse->Chase switch time (TPOINT_B in the MATLAB code).

    Chase starts at the last time point where the Pulse and Pulse&Chase ratios
    still agree; after that the chase drives them apart. If the two files are
    identical (or one is missing) we fall back to the middle measured time point.
    """
    t = protein_pulse.time
    # Compare on the shared, non-zero time points.
    a = protein_pulse.ratios
    b = protein_pandc.ratios
    n = min(a.shape[0], b.shape[0])
    if n == 0 or t.size < 3:
        return float(t[len(t) // 2]) if t.size else 14.0

    a, b = a[:n], b[:n]
    with np.errstate(invalid="ignore"):
        # per-time mean absolute difference across proteins
        diff = np.nanmean(np.abs(a - b), axis=0)
    # skip the prepended t=0 column
    idx = np.arange(1, t.size)
    agree = idx[diff[idx] < 1e-9]
    if agree.size:
        return float(t[agree[-1]])
    return float(t[len(t) // 2])
