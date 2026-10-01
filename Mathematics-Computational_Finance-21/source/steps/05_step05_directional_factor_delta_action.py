"""
Apply one directional factor of the source's scheme at substep s to a delta datum at cell k and return the minimum entry of the result. The factor must be applied by the single linear system the source prescribes for one application, at machine accuracy. Validates exact discrete mass conservation of the factor; excludes the 2-D composite scheme.

Below its positivity threshold the factor injects a signed transient, exactly as the exponential does below its eventual-positivity threshold; above the threshold the column is entrywise nonnegative. Benchmark values for this step: at s = 0.025 on the x generator with k = 21 the minimum entry is -2.05877060511e-05 (sub-threshold, negative); at s = 0.1 on the y generator with k = 9 it is -1.76658331498e-03; at the super-threshold substep s = 0.5 on the x generator with k = 21 it is +3.31113007043e-06 (entrywise positive). Every application sums to 1 to round-off because r(0) = 1 and the generator has zero column sums.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def directional_factor_delta_action(params: dict) -> float:
    """Minimum entry of one directional factor applied to a delta datum.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1, plus
        's' (substep length, > 0) and 'k' (1-based cell index of the
        delta datum, 1 <= k <= n). The factor is the source's
        second-order directional map at substep s; one application is
        computed by the single dense linear solve the source prescribes,
        applied to the unit vector e_k at machine accuracy.

    Returns
    -------
    float
        min entry of the resulting vector.

    Raises
    ------
    ValueError
        On invalid inputs, s <= 0, k out of range, or when the result
        does not sum to 1 within 1e-8 (mass conservation violated).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s05_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s05_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s05_grid(aL, aR, n)
    u = -kappa * (faces - m)
    L = np.zeros((n, n))
    for f in range(n - 1):
        uf = u[f]
        iL, iR = f, f + 1
        if uf >= 0.0:
            iu, iuu = iL, iL - 1
        else:
            iu, iuu = iR, iR + 1
        conv = {}
        if 0 <= iuu < n:
            conv[iu] = 1.5 * uf
            conv[iuu] = -0.5 * uf
        else:
            conv[iu] = uf
        for row, sgn in ((iL, -1.0), (iR, +1.0)):
            c = sgn / h
            for col, val in conv.items():
                L[row, col] += c * val
            L[row, iR] += c * (-sigma / h)
            L[row, iL] += c * (+sigma / h)
    return L, h


def _s05_validate1d(params, keys):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in keys:
        if k not in params:
            raise ValueError("missing key: " + k)
        if not np.isfinite(params[k]):
            raise ValueError("non-finite value for " + k)
    if params['n'] < 3 or int(params['n']) != params['n']:
        raise ValueError("n must be an integer >= 3")
    if params['aR'] <= params['aL']:
        raise ValueError("aR must exceed aL")
    if params['sigma'] <= 0 or params['kappa'] < 0:
        raise ValueError("sigma must be positive, kappa nonnegative")


def _oracle_directional_factor_delta_action(params):
    _s05_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma', 's', 'k'))
    n = int(params['n'])
    s, k = params['s'], int(params['k'])
    if s <= 0:
        raise ValueError("substep s must be positive")
    if not (1 <= k <= n):
        raise ValueError("cell index k out of range")
    L, _ = _s05_operator(params['aL'], params['aR'], n,
                         params['kappa'], params['m'], params['sigma'])
    M = np.eye(n) - s * L + (s * s / 2.0) * (L @ L)
    e = np.zeros(n)
    e[k - 1] = 1.0
    q = np.linalg.solve(M, e)
    if abs(q.sum() - 1.0) > 1e-8:
        raise ValueError("mass conservation violated by the factor")
    return float(q.min())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    p1 = dict({'aL': -2.4, 'aR': 2.4, 'n': 40, 'kappa': 4.2, 'm': 0.0, 'sigma': 1.0}, s=0.025, k=21)
    p2 = dict({'aL': -1.8, 'aR': 1.8, 'n': 10, 'kappa': 3.1, 'm': 0.0, 'sigma': 0.25}, s=0.1, k=9)
    p3 = dict({'aL': -2.4, 'aR': 2.4, 'n': 40, 'kappa': 4.2, 'm': 0.0, 'sigma': 1.0}, s=0.5, k=21)
    return [
        {"setup": "params = " + repr(p1),
         "call": "directional_factor_delta_action(params)",
         "gold_call": "_oracle_directional_factor_delta_action(params)"},
        {"setup": "params = " + repr(p2),
         "call": "directional_factor_delta_action(params)",
         "gold_call": "_oracle_directional_factor_delta_action(params)"},
        {"setup": "params = " + repr(p3),
         "call": "directional_factor_delta_action(params)",
         "gold_call": "_oracle_directional_factor_delta_action(params)"},
    ]
