"""
Assemble the 1-D flux-form directional Fokker-Planck generator on a cell-centred grid (zero-flux boundary faces, second-order linear-upwind convective face value with first-order upwind fallback at the boundary, central diffusive flux) and return the minimum off-diagonal entry of the assembled matrix, structural zeros included. Validates the conservative closure (column sums vanish to round-off) and the input parameters; the step deliberately excludes any time stepping.

The linear-upwind face value (3 p_u - p_uu)/2 contributes -u/(2h) on the second-upwind band whenever the face drift u is nonzero, so the generator carries genuinely negative off-diagonal entries at any nonzero drift while keeping zero column sums by flux telescoping. Benchmark values: the x-direction generator of the frozen configuration gives -37.80 exactly (= -0.5*9.072/0.12 from the face x = 2.16); the y-direction generator gives -4.65; a pure-diffusion operator (kappa = 0) has no negative off-diagonal entry and returns exactly 0.0.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def min_offdiagonal_entry(params: dict) -> float:
    """Minimum off-diagonal entry of the 1-D directional generator.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR' (domain bounds), 'n' (number of cells, integer
        >= 3), 'kappa' (drift rate, >= 0), 'm' (drift centre), 'sigma'
        (diffusion coefficient, > 0). The generator is the flux-form
        matrix pinned in the problem statement: (L p)_i =
        (J_{i-1/2} - J_{i+1/2})/h with zero boundary fluxes,
        J_{i+1/2} = u_{i+1/2}*phat - sigma*(p_{i+1}-p_i)/h, u(a) =
        -kappa*(a - m) at the face, and phat the second-order
        linear-upwind face value with first-order fallback where the
        second upwind cell does not exist.

    Returns
    -------
    float
        min over all entries L[i, j] with i != j (structural zeros
        included).

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


def _s01_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s01_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s01_grid(aL, aR, n)
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


def _s01_validate1d(params, keys):
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


def _oracle_min_offdiagonal_entry(params):
    _s01_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma'))
    L, _ = _s01_operator(params['aL'], params['aR'], int(params['n']),
                         params['kappa'], params['m'], params['sigma'])
    off = L.copy()
    np.fill_diagonal(off, 0.0)
    if np.abs(L.sum(axis=0)).max() > 1e-10 * np.abs(L).max():
        raise ValueError("conservative closure violated")
    return float(off.min())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    px = {'aL': -2.4, 'aR': 2.4, 'n': 40, 'kappa': 4.2, 'm': 0.0, 'sigma': 1.0}
    py = {'aL': -1.8, 'aR': 1.8, 'n': 10, 'kappa': 3.1, 'm': 0.0, 'sigma': 0.25}
    pe = {'aL': -1.0, 'aR': 1.0, 'n': 8, 'kappa': 0.0, 'm': 0.0, 'sigma': 0.3}
    return [
        {"setup": "params = " + repr(px),
         "call": "min_offdiagonal_entry(params)",
         "gold_call": "_oracle_min_offdiagonal_entry(params)"},
        {"setup": "params = " + repr(py),
         "call": "min_offdiagonal_entry(params)",
         "gold_call": "_oracle_min_offdiagonal_entry(params)"},
        {"setup": "params = " + repr(pe),
         "call": "min_offdiagonal_entry(params)",
         "gold_call": "_oracle_min_offdiagonal_entry(params)"},
    ]
