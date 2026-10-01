"""
Locate the positivity threshold that the source attaches to the first-order fallback factor of the directional sweep: form that factor's matrix function for the 1-D directional generator and report the edge of the one-sided window on which it is entrywise nonnegative under the pinned tolerance, located by the pinned bisection contract (bracket [glo, ghi], nbis iterations, final upper endpoint reported, 0.0 when the condition already holds at glo). Which matrix function belongs to the first-order factor is one of the ingredients the problem statement withholds and is not restated here; the second-order directional map is step 4.

For the eventually-positive (EM) generators of this benchmark the resolvent is entrywise nonnegative precisely on a one-sided window [gamma0, infinity): refining the substep at fixed mesh destroys the sign. Benchmark values for this step: gamma0 = 0.00887115196684 for the x generator and 0.06990006290861 for the y generator; a Metzler generator (pure diffusion, kappa = 0) has a nonnegative resolvent for every positive substep, and the pinned procedure returns exactly 0.0.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def first_order_threshold(params: dict) -> float:
    """Positivity threshold of the first-order factor's matrix function.

    Parameters
    ----------
    params : dict
        Keys 'aL', 'aR', 'n', 'kappa', 'm', 'sigma' as in step 1, plus
        'glo', 'ghi' (bisection bracket endpoints, 0 < glo < ghi),
        'tol_pos' (entrywise-nonnegativity tolerance; an entry counts as
        nonnegative when >= tol_pos), and 'nbis' (bisection iterations).
        The matrix function is the one the source attaches to the
        first-order fallback factor of the directional sweep; it is
        formed densely and inverted at machine accuracy, and the
        entrywise minimum runs over all entries.

    Returns
    -------
    float
        0.0 when the entrywise minimum at glo already clears tol_pos;
        otherwise the final upper bisection endpoint after nbis
        iterations on [glo, ghi].

    Raises
    ------
    ValueError
        On invalid inputs, or when the entrywise minimum at ghi does not
        clear tol_pos (no window inside the bracket).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s03_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s03_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s03_grid(aL, aR, n)
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


def _s03_validate1d(params, keys):
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

def _s03_bisect(minfun, glo, ghi, tol_pos, nbis):
    """Pinned threshold location: return 0.0 when the condition already
    holds at glo; raise when it fails at ghi; otherwise 200-step bisection
    keeping a violating / b satisfying, reporting the final b."""
    if minfun(glo) >= tol_pos:
        return 0.0
    if not (minfun(ghi) >= tol_pos):
        raise ValueError("no positivity window inside the bracket")
    a, b = glo, ghi
    for _ in range(int(nbis)):
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        if minfun(mid) >= tol_pos:
            b = mid
        else:
            a = mid
    return float(b)


def _oracle_first_order_threshold(params):
    _s03_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma',
                             'glo', 'ghi', 'tol_pos', 'nbis'))
    if not (0 < params['glo'] < params['ghi']):
        raise ValueError("need 0 < glo < ghi")
    n = int(params['n'])
    L, _ = _s03_operator(params['aL'], params['aR'], n,
                         params['kappa'], params['m'], params['sigma'])
    I = np.eye(n)

    def minfun(g):
        return np.linalg.inv(I - g * L).min()

    return _s03_bisect(minfun, params['glo'], params['ghi'],
                       params['tol_pos'], params['nbis'])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    thr = {'glo': 1e-6, 'ghi': 64.0, 'tol_pos': -1e-12, 'nbis': 200}
    px = dict({'aL': -2.4, 'aR': 2.4, 'n': 40, 'kappa': 4.2, 'm': 0.0, 'sigma': 1.0}, **thr)
    py = dict({'aL': -1.8, 'aR': 1.8, 'n': 10, 'kappa': 3.1, 'm': 0.0, 'sigma': 0.25}, **thr)
    pe = dict({'aL': -1.0, 'aR': 1.0, 'n': 8, 'kappa': 0.0, 'm': 0.0, 'sigma': 0.3}, **thr)
    return [
        {"setup": "params = " + repr(px),
         "call": "first_order_threshold(params)",
         "gold_call": "_oracle_first_order_threshold(params)"},
        {"setup": "params = " + repr(py),
         "call": "first_order_threshold(params)",
         "gold_call": "_oracle_first_order_threshold(params)"},
        {"setup": "params = " + repr(pe),
         "call": "first_order_threshold(params)",
         "gold_call": "_oracle_first_order_threshold(params)"},
    ]
