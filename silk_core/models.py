"""
Kinetic prediction models for protein labelling (Hpoi).

Two models are provided, both re-implemented directly from the MATLAB objective
functions of the SILK toolbox:

* ``single_exponential``  -- "single exponential pool model" (F400_def_Obj2 /
  F204).  The precursor is assumed to label instantly, so the labelled fraction
  of a protein is ``LR(t) = 1 - exp(-k t)`` during the pulse, decaying as
  ``LR(tp) * exp(-k (t-tp))`` during the chase.  The fitted parameter is the
  turnover **rate** ``k`` (per day); half-life = ln(2) / k.

* ``two_compartment``    -- "2-compartment model" (F300_def_Obj2 / F203).  The
  protein is driven by a bi-exponential soluble precursor characterised by
  ``(T1, T2, A)`` for the pulse and ``(A1, A2)`` for the chase.  The fitted
  parameter is the turnover **time constant** ``tau`` (days); half-life =
  ln(2) * tau.

Both functions take a full time vector (including the prepended t = 0) and the
Pulse->Chase switch time ``tpoint``.  For a pure pulse experiment pass a
``tpoint`` larger than the last time point so the chase branch is never taken.
"""

from __future__ import annotations

import numpy as np

LN2 = np.log(2.0)
_EXP_CLIP = 700.0          # keep exp() arguments inside float64 range
_SING_EPS = 1e-6           # threshold for the removable T == tau singularity


# --------------------------------------------------------------------------- #
#  half-life <-> parameter conversions
# --------------------------------------------------------------------------- #
def rate_to_halflife(k: float) -> float:
    """single-exponential: k (rate, /day) -> half-life (days)."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return LN2 / k


def timeconstant_to_halflife(tau: float) -> float:
    """2-compartment: tau (time constant, days) -> half-life (days)."""
    return LN2 * np.asarray(tau, dtype=float)


# --------------------------------------------------------------------------- #
#  single exponential pool model
# --------------------------------------------------------------------------- #
def single_exponential(k: float, t: np.ndarray, tpoint: float, chase: bool) -> np.ndarray:
    """
    Predicted Label/Total ratio for the single-exponential pool model.

    Parameters
    ----------
    k       : turnover rate (per day), the fitted parameter.
    t       : time vector (days).
    tpoint  : Pulse->Chase switch time.
    chase   : if True this is Pulse&Chase data (apply the chase branch for
              t > tpoint); if False it is pure Pulse data.
    """
    t = np.asarray(t, dtype=float)
    k = float(k)
    with np.errstate(over="ignore", invalid="ignore"):
        pulse = 1.0 - np.exp(np.clip(-k * t, -_EXP_CLIP, _EXP_CLIP))
        if not chase:
            return pulse
        # chase: label at tpoint decays exponentially afterwards
        btp = 1.0 - np.exp(np.clip(-k * tpoint, -_EXP_CLIP, _EXP_CLIP))
        s = t - tpoint
        chase_vals = btp * np.exp(np.clip(-k * s, -_EXP_CLIP, _EXP_CLIP))
        return np.where(t > tpoint, chase_vals, pulse)


# --------------------------------------------------------------------------- #
#  2-compartment model
# --------------------------------------------------------------------------- #
def _comp_term(t: np.ndarray, tau: float, T: float) -> np.ndarray:
    """
    Vectorised ``T * (1 - exp(t/tau - t/T)) / (T - tau)`` with the removable
    singularity at ``tau == T`` handled analytically (limit = -t / T).
    """
    t = np.asarray(t, dtype=float)
    den = T - tau
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expo = np.clip(t / tau - t / T, -_EXP_CLIP, _EXP_CLIP)
        val = T * (1.0 - np.exp(expo)) / den
    # analytic limit where T ~= tau
    limit = -t / T
    return np.where(np.abs(den) < _SING_EPS, limit, val)


def two_compartment(
    tau: float,
    t: np.ndarray,
    tpoint: float,
    chase: bool,
    precursor: dict,
) -> np.ndarray:
    """
    Predicted Label/Total ratio for the 2-compartment model.

    Parameters
    ----------
    tau       : protein turnover time constant (days), the fitted parameter.
    t         : time vector (days).
    tpoint    : Pulse->Chase switch time.
    chase     : apply the chase branch for t > tpoint when True.
    precursor : dict with keys T1, T2, A (pulse) and, for the chase, A1, A2.
    """
    t = np.asarray(t, dtype=float)
    tau = float(tau)
    T1 = float(precursor["T1"])
    T2 = float(precursor["T2"])
    A = float(precursor["A"])

    with np.errstate(over="ignore", invalid="ignore"):
        # --- pulse branch (valid for all t) ---
        b1 = A * _comp_term(t, tau, T1)
        b2 = (1.0 - A) * _comp_term(t, tau, T2)
        pulse = 1.0 - np.exp(np.clip(-t / tau, -_EXP_CLIP, _EXP_CLIP)) * (1.0 - b1 - b2)

        if not chase:
            return pulse

        # --- chase branch (t > tpoint) ---
        A1 = float(precursor.get("A1", np.nan))
        A2 = float(precursor.get("A2", np.nan))
        # labelled fraction reached at the switch time
        b1_tp = A * _comp_term(np.array([tpoint]), tau, T1)[0]
        b2_tp = (1.0 - A) * _comp_term(np.array([tpoint]), tau, T2)[0]
        btp = 1.0 - np.exp(np.clip(-tpoint / tau, -_EXP_CLIP, _EXP_CLIP)) * (1.0 - b1_tp - b2_tp)
        if not np.isfinite(btp):
            btp = 0.0

        s = t - tpoint
        cb1 = A1 * _comp_term(s, tau, T1)
        cb2 = A2 * _comp_term(s, tau, T2)
        chase_vals = np.exp(np.clip(-s / tau, -_EXP_CLIP, _EXP_CLIP)) * (btp - cb1 - cb2)
        return np.where(t > tpoint, chase_vals, pulse)


# --------------------------------------------------------------------------- #
#  unified dispatcher
# --------------------------------------------------------------------------- #
def predict(param, t, tpoint, chase, model="single", precursor=None):
    """
    Dispatch to the requested model.

    ``model`` is 'single' or '2comp'.  ``param`` is k (single) or tau (2comp).
    """
    if model in ("single", "single_exponential"):
        return single_exponential(param, t, tpoint, chase)
    if model in ("2comp", "two_compartment"):
        if precursor is None:
            raise ValueError("2-compartment model requires precursor parameters.")
        return two_compartment(param, t, tpoint, chase, precursor)
    raise ValueError(f"Unknown model: {model!r}")


def param_to_halflife(param, model="single"):
    if model in ("single", "single_exponential"):
        return rate_to_halflife(param)
    if model in ("2comp", "two_compartment"):
        return timeconstant_to_halflife(param)
    raise ValueError(f"Unknown model: {model!r}")


def halflife_to_param(halflife, model="single"):
    """Inverse of :func:`param_to_halflife` (used for initial guesses)."""
    halflife = np.asarray(halflife, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if model in ("single", "single_exponential"):
            return LN2 / halflife            # k
        if model in ("2comp", "two_compartment"):
            return halflife / LN2            # tau
    raise ValueError(f"Unknown model: {model!r}")


# --------------------------------------------------------------------------- #
#  vectorised grid predictors  -- evaluate a whole (G,) parameter grid at once
#  over the (T,) time vector, returning a (G, T) matrix. Used to build the
#  per-protein fitting grid instantly (and to keep the global precursor search
#  fast, since it rebuilds the grid on every outer-optimisation step).
# --------------------------------------------------------------------------- #
def _comp_term_grid(t, tau, T):
    """(G,T) version of ``T*(1 - exp(t/tau - t/T))/(T - tau)`` with the
    removable singularity at ``tau == T`` handled (limit = -t/T)."""
    tt = np.asarray(t, dtype=float)[None, :]      # (1,T)
    ta = np.asarray(tau, dtype=float)[:, None]    # (G,1)
    den = T - ta                                  # (G,1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expo = np.clip(tt / ta - tt / T, -_EXP_CLIP, _EXP_CLIP)
        val = T * (1.0 - np.exp(expo)) / den      # (G,T)
    limit = np.broadcast_to(-tt / T, val.shape)
    return np.where(np.abs(den) < _SING_EPS, limit, val)


def single_exponential_grid(k, t, tpoint, chase):
    k = np.asarray(k, dtype=float)[:, None]       # (G,1)
    tt = np.asarray(t, dtype=float)[None, :]      # (1,T)
    with np.errstate(over="ignore", invalid="ignore"):
        pulse = 1.0 - np.exp(np.clip(-k * tt, -_EXP_CLIP, _EXP_CLIP))
        if not chase:
            return pulse
        btp = 1.0 - np.exp(np.clip(-k * tpoint, -_EXP_CLIP, _EXP_CLIP))   # (G,1)
        s = tt - tpoint
        chase_vals = btp * np.exp(np.clip(-k * s, -_EXP_CLIP, _EXP_CLIP))
        return np.where(tt > tpoint, chase_vals, pulse)


def two_compartment_grid(tau, t, tpoint, chase, precursor):
    tau = np.asarray(tau, dtype=float)
    tt = np.asarray(t, dtype=float)[None, :]      # (1,T)
    ta = tau[:, None]                             # (G,1)
    T1, T2, A = float(precursor["T1"]), float(precursor["T2"]), float(precursor["A"])
    with np.errstate(over="ignore", invalid="ignore"):
        b1 = A * _comp_term_grid(t, tau, T1)
        b2 = (1.0 - A) * _comp_term_grid(t, tau, T2)
        pulse = 1.0 - np.exp(np.clip(-tt / ta, -_EXP_CLIP, _EXP_CLIP)) * (1.0 - b1 - b2)
        if not chase:
            return pulse

        A1, A2 = float(precursor.get("A1", np.nan)), float(precursor.get("A2", np.nan))
        b1tp = A * _comp_term_grid([tpoint], tau, T1)[:, 0]
        b2tp = (1.0 - A) * _comp_term_grid([tpoint], tau, T2)[:, 0]
        btp = 1.0 - np.exp(np.clip(-tpoint / tau, -_EXP_CLIP, _EXP_CLIP)) * (1.0 - b1tp - b2tp)
        btp = np.where(np.isfinite(btp), btp, 0.0)[:, None]
        s = tt - tpoint
        cb1 = A1 * _comp_term_grid(np.asarray(t, dtype=float) - tpoint, tau, T1)
        cb2 = A2 * _comp_term_grid(np.asarray(t, dtype=float) - tpoint, tau, T2)
        chase_vals = np.exp(np.clip(-s / ta, -_EXP_CLIP, _EXP_CLIP)) * (btp - cb1 - cb2)
        return np.where(tt > tpoint, chase_vals, pulse)


def predict_grid(params, t, tpoint, chase, model="single", precursor=None):
    """Vectorised dispatcher -> (len(params), len(t)) prediction matrix."""
    if model in ("single", "single_exponential"):
        return single_exponential_grid(params, t, tpoint, chase)
    if model in ("2comp", "two_compartment"):
        if precursor is None:
            raise ValueError("2-compartment model requires precursor parameters.")
        return two_compartment_grid(params, t, tpoint, chase, precursor)
    raise ValueError(f"Unknown model: {model!r}")
