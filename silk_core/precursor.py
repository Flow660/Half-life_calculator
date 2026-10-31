"""
Fit the soluble precursor (Hsol) labelling curve.

Pulse model (matches the MATLAB constraint function F131/F132):

    Hsol_pulse(t) = 1 - A*exp(-t/T1) - (1-A)*exp(-t/T2)

which rises from 0 towards a plateau.  Fitted parameters: T1, T2 (day-scale time
constants, T1 < T2) and A (mixing fraction in (0, 1)).

Chase model (Pulse&Chase precursor, after the switch time ``tpoint``):

    Hsol_chase(s) = A1*exp(-s/T1) + A2*exp(-s/T2),   s = t - tpoint

with T1, T2 carried over from the pulse fit and A1, A2 obtained by linear least
squares.  Continuity gives A1 + A2 ~= Hsol_pulse(tpoint).

These parameters (T1, T2, A, A1, A2) are exactly the ``xfinal_a`` vector the
MATLAB 2-compartment Hpoi model consumes.
"""

from __future__ import annotations

import numpy as np

from . import _optim

try:                                   # optional, only used if present
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _HAVE_SCIPY = True
except Exception:                      # pragma: no cover
    _HAVE_SCIPY = False


def _pulse_curve(t, T1, T2, A):
    t = np.asarray(t, dtype=float)
    return 1.0 - A * np.exp(-t / T1) - (1.0 - A) * np.exp(-t / T2)


def _unpack(p):
    """Unconstrained params -> (T1, T2, A) with T1<T2>0 and 0<A<1."""
    T1 = np.exp(p[0])
    T2 = T1 + np.exp(p[1])
    A = 1.0 / (1.0 + np.exp(-p[2]))
    return T1, T2, A


def fit_precursor(pulse, pandc=None, tpoint=None):
    """
    Fit the precursor.

    Parameters
    ----------
    pulse  : PrecursorData for the Hsol Pulse file (required).
    pandc  : PrecursorData for the Hsol Pulse&Chase file (optional).
    tpoint : chase switch time.  If None it is inferred from the peak of the
             Pulse&Chase curve (falling edge marks the chase).

    Returns
    -------
    dict with keys T1, T2, A, A1, A2, plus diagnostics r2_pulse, r2_chase,
    tpoint, and the fitted curves for plotting.
    """
    t = np.asarray(pulse.time, dtype=float)
    y = np.asarray(pulse.ratio, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t_f, y_f = t[mask], y[mask]

    # scale-aware initial guesses from the data span
    tmax = max(float(np.nanmax(t_f)), 1.0)

    def sse(p):
        T1, T2, A = _unpack(p)
        pred = _pulse_curve(t_f, T1, T2, A)
        r = pred - y_f
        if not np.all(np.isfinite(r)):
            return 1e18
        return float(np.sum(r * r))

    best = None
    for T1_0 in (tmax * 0.05, tmax * 0.2, tmax * 0.5, 2.0):
        for T2_0 in (tmax * 0.5, tmax * 2.0, tmax * 10.0):
            if T2_0 <= T1_0:
                T2_0 = T1_0 * 3.0
            for A0 in (0.3, 0.5, 0.7):
                p0 = np.array([np.log(T1_0), np.log(max(T2_0 - T1_0, 1e-3)),
                               np.log(A0 / (1.0 - A0))])
                x, f = _optim.nelder_mead(sse, p0, step=0.5)
                if best is None or f < best[1]:
                    best = (x, f)

    T1, T2, A = _unpack(best[0])

    # optional SciPy polish (better optimum + honours simple bounds)
    if _HAVE_SCIPY:
        try:
            popt, _ = _scipy_curve_fit(
                _pulse_curve, t_f, y_f, p0=[T1, T2, A],
                bounds=([1e-3, 1e-3, 0.0], [1e5, 1e6, 1.0]), maxfev=20000,
            )
            if _pulse_sse(popt, t_f, y_f) <= best[1] * 1.0001:
                T1, T2, A = float(popt[0]), float(popt[1]), float(popt[2])
                if T1 > T2:
                    T1, T2 = T2, T1
                    A = 1.0 - A
        except Exception:
            pass

    r2_pulse = _r2(y_f, _pulse_curve(t_f, T1, T2, A))

    result = {
        "T1": float(T1), "T2": float(T2), "A": float(A),
        "A1": float("nan"), "A2": float("nan"),
        "r2_pulse": float(r2_pulse), "r2_chase": float("nan"),
        "tpoint": None,
        "pulse_data": (t_f, y_f),
        "chase_data": None,
    }

    # ---- chase amplitudes A1, A2 (linear least squares) ----
    if pandc is not None:
        tc = np.asarray(pandc.time, dtype=float)
        yc = np.asarray(pandc.ratio, dtype=float)
        m = np.isfinite(tc) & np.isfinite(yc)
        tc, yc = tc[m], yc[m]

        if tpoint is None:
            # falling edge: last time before the curve starts to drop = peak time
            if yc.size >= 2:
                tpoint = float(tc[int(np.argmax(yc))])
            else:
                tpoint = float(tmax / 2.0)
        result["tpoint"] = float(tpoint)

        chase_mask = tc > tpoint
        if chase_mask.sum() >= 1:
            s = tc[chase_mask] - tpoint
            obs = yc[chase_mask]
            basis = np.column_stack([np.exp(-s / T1), np.exp(-s / T2)])
            try:
                coef, *_ = np.linalg.lstsq(basis, obs, rcond=None)
                A1, A2 = float(coef[0]), float(coef[1])
            except Exception:
                # fall back to continuity split
                S = _pulse_curve(np.array([tpoint]), T1, T2, A)[0]
                A1, A2 = S, 0.0
            result["A1"], result["A2"] = A1, A2
            result["r2_chase"] = _r2(obs, basis @ np.array([A1, A2]))
            result["chase_data"] = (tc[chase_mask], obs)

    return result


def estimate_precursor_global(t, pulse_ratios, pandc_ratios=None, tpoint=14.0,
                              fit_pandc=False, max_proteins=None, n_starts=4,
                              progress=None):
    """
    Paper-style precursor estimation (Fornasiero et al. 2018, Supplementary Fig. 3).

    Instead of fitting a measured plasma curve, the soluble-pool parameters
    {T1, T2, A} (same double-exponential form as the plasma fit) are obtained by
    globally fitting the labelling dynamics of *all reliably-detected proteins*:
    for a candidate precursor, every protein is given its best-fit turnover and
    the total sum-of-squared errors is minimised over {T1, T2, A}.  The chase
    amplitudes {A1, A2} are then obtained by linear least squares against the
    Pulse&Chase data (they enter the chase prediction linearly).

    No plasma data is used.

    Parameters
    ----------
    t             : time vector (T,), includes prepended 0.
    pulse_ratios  : (N, T) protein Pulse ratios.
    pandc_ratios  : (N, T) protein Pulse&Chase ratios (needed for A1,A2).
    tpoint        : Pulse->Chase switch (days).
    fit_pandc     : also estimate chase amplitudes {A1, A2}.

    Returns a dict compatible with the plasma-fit result (T1,T2,A,A1,A2,tpoint,...).
    """
    from . import fitting  # local import to avoid a cycle

    t = np.asarray(t, dtype=float)
    op = np.asarray(pulse_ratios, dtype=float)

    # reliably-detected proteins: finite at every measured (non-zero) time point
    meas = t > 0
    reliable = np.isfinite(op[:, meas]).all(axis=1)
    idx = np.where(reliable)[0]
    if max_proteins is not None and idx.size > max_proteins:
        # even, deterministic subsample (no RNG in this environment)
        step = int(np.ceil(idx.size / max_proteins))
        idx = idx[::step]
    op_sub = op[idx]
    n_used = idx.size

    # keep the outer search fast: use a representative (deterministic) subsample
    OUTER_MAX = 1500
    if op_sub.shape[0] > OUTER_MAX:
        step = int(np.ceil(op_sub.shape[0] / OUTER_MAX))
        op_out = op_sub[::step]
    else:
        op_out = op_sub

    # physically-motivated bounds for the soluble-pool double-exponential
    T1_LO, T1_HI = 0.3, 25.0
    T2_LO, T2_HI = 10.0, 400.0
    A_LO, A_HI = 0.02, 0.90

    def sse(T1, T2, A):
        try:
            return fitting.total_sse_for_precursor(
                t, op_out, None, tpoint, "2comp",
                {"T1": T1, "T2": T2, "A": A}, fit_pandc=False, n_grid=120)
        except Exception:
            return 1e18

    # ---- coarse grid surface scan (reproduces the paper's SSE surface) ----
    T1_grid = np.geomspace(T1_LO, T1_HI, 11)
    T2_grid = np.geomspace(T2_LO, T2_HI, 10)
    A_grid = np.linspace(A_LO, A_HI, 8)
    best = (np.inf, T1_grid[0], T2_grid[0], A_grid[0])
    total = len(T1_grid) * len(T2_grid) * len(A_grid)
    done = 0
    for T1c in T1_grid:
        for T2c in T2_grid:
            if T2c <= T1c:
                continue
            for Ac in A_grid:
                s = sse(T1c, T2c, Ac)
                if s < best[0]:
                    best = (s, T1c, T2c, Ac)
            done += len(A_grid)
        if progress is not None:
            progress(done, total)

    # ---- local refinement around the coarse minimum ----
    _, T1b, T2b, Ab = best
    for _ in range(2):
        T1r = np.linspace(max(T1b * 0.6, T1_LO), min(T1b * 1.6, T1_HI), 7)
        T2r = np.linspace(max(T2b * 0.6, T2_LO), min(T2b * 1.6, T2_HI), 7)
        Ar = np.linspace(max(Ab - 0.08, A_LO), min(Ab + 0.08, A_HI), 7)
        for T1c in T1r:
            for T2c in T2r:
                if T2c <= T1c:
                    continue
                for Ac in Ar:
                    s = sse(T1c, T2c, Ac)
                    if s < best[0]:
                        best = (s, T1c, T2c, Ac)
        _, T1b, T2b, Ab = best

    T1, T2, A = float(T1b), float(T2b), float(Ab)

    # flag when the optimum sits on a bound (weakly identified precursor)
    railed = (
        T1 <= T1_LO * 1.05 or T1 >= T1_HI * 0.95
        or T2 <= T2_LO * 1.05 or T2 >= T2_HI * 0.95
    )

    result = {
        "T1": T1, "T2": T2, "A": A,
        "A1": float("nan"), "A2": float("nan"),
        "r2_pulse": float("nan"), "r2_chase": float("nan"),
        "tpoint": float(tpoint), "method": "global",
        "n_proteins_used": int(n_used), "global_sse": float(best[0]),
        "railed": bool(railed),
        "pulse_data": None, "chase_data": None,
    }

    # ---- chase amplitudes {A1, A2} via linear least squares ----
    # Only attempt when the pulse precursor is well-identified; a railed precursor
    # makes the linear chase fit blow up (negative / non-physical amplitudes).
    if fit_pandc and pandc_ratios is not None and not railed:
        oc = np.asarray(pandc_ratios, dtype=float)[idx]
        prec_p = {"T1": T1, "T2": T2, "A": A}
        # per-protein turnover from the pulse fit under this precursor
        fr = fitting.fit_batch(t, op_sub, None, tpoint, "2comp", prec_p,
                               fit_pandc=False, labels=None)
        tau = np.array([r.parameter for r in fr], dtype=float)
        good = np.isfinite(tau)
        tau_g = tau[good]
        oc_g = oc[good]

        tc = t[t > tpoint]
        if tc.size and tau_g.size:
            s = tc - tpoint
            from . import models as _m
            btp = _m.two_compartment_grid(tau_g, [tpoint], tpoint, False, prec_p)[:, 0]  # (n,)
            E = np.exp(np.clip(-s[None, :] / tau_g[:, None], -700, 700))                 # (n,S)
            ct1 = _m._comp_term_grid(s, tau_g, T1)
            ct2 = _m._comp_term_grid(s, tau_g, T2)
            X1 = (E * ct1).ravel()
            X2 = (E * ct2).ravel()
            C = (E * btp[:, None]).ravel()
            obs = oc_g[:, [i for i, tv in enumerate(t) if tv > tpoint]].ravel()
            y = C - obs
            m = np.isfinite(y) & np.isfinite(X1) & np.isfinite(X2)
            if m.sum() >= 2:
                Amat = np.column_stack([X1[m], X2[m]])
                coef, *_ = np.linalg.lstsq(Amat, y[m], rcond=None)
                result["A1"], result["A2"] = float(coef[0]), float(coef[1])

    # Physical fallback for the chase amplitudes when the linear fit is
    # unavailable or non-physical (negative): split the pulse plateau at tpoint
    # by continuity (A1 + A2 = Hsol_pulse(tpoint)), weighted like the pulse.
    if fit_pandc:
        A1v, A2v = result["A1"], result["A2"]
        if (not np.isfinite(A1v) or not np.isfinite(A2v) or A1v < 0 or A2v < 0):
            S = float(_pulse_curve(np.array([tpoint]), T1, T2, A)[0])
            result["A1"], result["A2"] = A * S, (1.0 - A) * S
            result["chase_amplitudes_fallback"] = True

    return result


def _pulse_sse(popt, t_f, y_f):
    pred = _pulse_curve(t_f, *popt)
    r = pred - y_f
    return float(np.sum(r * r)) if np.all(np.isfinite(r)) else 1e18


def _r2(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(obs) & np.isfinite(pred)
    if m.sum() < 2:
        return float("nan")
    ss_res = np.sum((obs[m] - pred[m]) ** 2)
    ss_tot = np.sum((obs[m] - np.mean(obs[m])) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def precursor_curve(params, t, chase=False):
    """Evaluate the fitted precursor curve for plotting."""
    t = np.asarray(t, dtype=float)
    T1, T2, A = params["T1"], params["T2"], params["A"]
    if not chase or params.get("tpoint") is None or not np.isfinite(params.get("A1", np.nan)):
        return _pulse_curve(t, T1, T2, A)
    tp = params["tpoint"]
    A1, A2 = params["A1"], params["A2"]
    out = _pulse_curve(t, T1, T2, A)
    s = t - tp
    chase_vals = A1 * np.exp(-s / T1) + A2 * np.exp(-s / T2)
    return np.where(t > tp, chase_vals, out)
