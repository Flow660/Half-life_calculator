"""
Small, dependency-free optimizers.

Kept deliberately tiny so the SILK core runs on numpy alone.  When SciPy is
installed the higher-level modules use it for tighter results and covariance
estimates, but everything here is enough to fit precursors and per-protein
turnover without any extra dependency.
"""

from __future__ import annotations

import numpy as np

_GOLD = 0.3819660112501051  # 2 - golden ratio


def minimize_scalar_bounded(func, lo, hi, xtol=1e-7, maxiter=200):
    """
    Golden-section + parabolic-ish search for the minimum of a 1-D function on
    ``[lo, hi]``.  Robust for the smooth, unimodal SSE(param) curves that arise
    when fitting a single turnover parameter.

    Returns ``(x_min, f_min)``.
    """
    lo, hi = float(lo), float(hi)
    a, b = lo, hi
    x = a + _GOLD * (b - a)
    w = v = x
    fx = fw = fv = func(x)
    d = e = 0.0

    for _ in range(maxiter):
        m = 0.5 * (a + b)
        tol1 = xtol * abs(x) + 1e-12
        tol2 = 2.0 * tol1
        if abs(x - m) <= tol2 - 0.5 * (b - a):
            break

        use_golden = True
        if abs(e) > tol1:
            # try a parabolic step
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if not (abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x)):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < m else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < m else (a - x)
            d = _GOLD * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = func(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


def nelder_mead(func, x0, step=0.5, xtol=1e-6, ftol=1e-8, maxiter=2000):
    """
    Classic Nelder-Mead simplex minimizer (unconstrained).  Returns
    ``(x_best, f_best)``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    # build initial simplex
    sim = np.empty((n + 1, n))
    sim[0] = x0
    for i in range(n):
        y = x0.copy()
        y[i] += step if y[i] == 0 else step * (1.0 + abs(y[i]))
        sim[i + 1] = y
    f = np.array([func(x) for x in sim])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(f)
        sim, f = sim[order], f[order]
        if (np.max(np.abs(sim[1:] - sim[0])) <= xtol) and (abs(f[-1] - f[0]) <= ftol):
            break
        centroid = sim[:-1].mean(axis=0)
        # reflection
        xr = centroid + alpha * (centroid - sim[-1])
        fr = func(xr)
        if f[0] <= fr < f[-2]:
            sim[-1], f[-1] = xr, fr
            continue
        if fr < f[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                sim[-1], f[-1] = xe, fe
            else:
                sim[-1], f[-1] = xr, fr
            continue
        # contraction
        xc = centroid + rho * (sim[-1] - centroid)
        fc = func(xc)
        if fc < f[-1]:
            sim[-1], f[-1] = xc, fc
            continue
        # shrink
        sim[1:] = sim[0] + sigma * (sim[1:] - sim[0])
        for i in range(1, n + 1):
            f[i] = func(sim[i])
    order = np.argsort(f)
    return sim[order][0], f[order][0]
