"""
Compute the discrete stationary density of the 1-D directional generator (the kernel vector of L normalized so its entries sum to 1, obtained by a machine-accuracy direct solve) and return its smallest entry. Validates the normalization; excludes any time stepping.

The generator is irreducible with a simple zero eigenvalue, so the stationary vector is unique up to scale; its strict positivity is the Perron structure that eventual positivity of the semigroup requires. Benchmark values for this step: minimum stationary mass 4.84540506440e-06 for the x generator, 3.38909218799e-04 for the y generator, and exactly 1/n = 0.125 for the pure-diffusion edge case kappa = 0, n = 8, whose stationary density is uniform.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def stationary_minimum_mass(params: dict) -> float:
    """Smallest entry of the normalized discrete stationary density.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1; the
        generator is the same pinned flux-form matrix.

    Returns
    -------
    float
        min_k p_inf[k] where L p_inf = 0 and sum_k p_inf[k] = 1 (no
        cell-width weighting).

    Raises
    ------
    ValueError
        If a key is missing or non-finite, n < 3, aR <= aL, sigma <= 0,
        or kappa < 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s02_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s02_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s02_grid(aL, aR, n)
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


def _s02_validate1d(params, keys):
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


def _oracle_stationary_minimum_mass(params):
    _s02_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma'))
    n = int(params['n'])
    L, _ = _s02_operator(params['aL'], params['aR'], n,
                         params['kappa'], params['m'], params['sigma'])
    M = L.copy()
    M[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    p = np.linalg.solve(M, b)
    if abs(p.sum() - 1.0) > 1e-8:
        raise ValueError("stationary normalization failed")
    return float(p.min())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    px = {'aL': -2.4, 'aR': 2.4, 'n': 40, 'kappa': 4.2, 'm': 0.0, 'sigma': 1.0}
    py = {'aL': -1.8, 'aR': 1.8, 'n': 10, 'kappa': 3.1, 'm': 0.0, 'sigma': 0.25}
    pe = {'aL': -1.0, 'aR': 1.0, 'n': 8, 'kappa': 0.0, 'm': 0.0, 'sigma': 0.3}
    return [
        {"setup": "params = " + repr(px),
         "call": "stationary_minimum_mass(params)",
         "gold_call": "_oracle_stationary_minimum_mass(params)"},
        {"setup": "params = " + repr(py),
         "call": "stationary_minimum_mass(params)",
         "gold_call": "_oracle_stationary_minimum_mass(params)"},
        {"setup": "params = " + repr(pe),
         "call": "stationary_minimum_mass(params)",
         "gold_call": "_oracle_stationary_minimum_mass(params)"},
    ]
