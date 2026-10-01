"""
Assemble the conservative one-sided mixed operator Axy = 2*Sxy*kron(Dx, Dy) (face-form one-sided differences with zero boundary faces; forward-biased in x and backward-biased in y for Sxy >= 0, forward-biased in both directions for Sxy < 0), build the pinned initial datum, and return the entry of Axy p0 at the query cell (iq, jq). Validates the exact zero column sums of the closure; excludes time stepping.

The face-form closure keeps every column sum of Axy at zero to round-off, so the mixed factor conserves discrete mass exactly, and the one-sided pairing keeps the spectrum of Axy real and non-positive - the property that makes an implicit treatment of the mixed block stable. Benchmark values for this step, evaluated at the spike cell (21, 6) of the frozen datum: -169.075246342 for rho = +0.4 and -168.515008336 for rho = -0.4 (the orientation change is numerically visible), and exactly 0.0 for rho = 0.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mixed_action_at_node(params: dict) -> float:
    """Entry of Axy p0 at one query cell for the pinned mixed operator.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys 'xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
        'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho', 'x0', 'y0', 'wx',
        'wy', 'isp', 'jsp', 'fsp' as pinned in the problem statement,
        plus the query cell 'iq', 'jq' (1-based). The datum is the
        normalized Gaussian-plus-spike vector with x-major ordering
        ell = (i-1)*ny + j.

    Returns
    -------
    float
        (Axy @ p0)[(iq-1)*ny + (jq-1)] in 0-based storage.

    Raises
    ------
    ValueError
        On invalid inputs, a query cell out of range, or a violated
        conservative closure.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s06_grid_centers(aL, aR, n):
    h = (aR - aL) / n
    return h, aL + (np.arange(1, n + 1) - 0.5) * h


def _s06_oneside_D(n, h, side):
    """Conservative face-form one-sided difference: (D p)_i =
    (g_{i+1/2} - g_{i-1/2})/h with g = 0 at the two boundary faces.
    side 'F': g_{i+1/2} = (3 p_{i+1} - p_{i+2})/2, fallback p_{i+1};
    side 'B': g_{i+1/2} = (3 p_i - p_{i-1})/2, fallback p_i."""
    G = np.zeros((n - 1, n))
    for f in range(n - 1):
        if side == 'F':
            if f + 2 < n:
                G[f, f + 1] = 1.5
                G[f, f + 2] = -0.5
            else:
                G[f, f + 1] = 1.0
        else:
            if f - 1 >= 0:
                G[f, f] = 1.5
                G[f, f - 1] = -0.5
            else:
                G[f, f] = 1.0
    D = np.zeros((n, n))
    for i in range(n):
        if i < n - 1:
            D[i, :] += G[i, :] / h
        if i > 0:
            D[i, :] -= G[i - 1, :] / h
    return D


def _s06_mixed(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx = (params['xR'] - params['xL']) / nx
    hy = (params['yR'] - params['yL']) / ny
    Sxy = params['rho'] * np.sqrt(params['Sxx'] * params['Syy'])
    if Sxy >= 0:
        Dx = _s06_oneside_D(nx, hx, 'F')
        Dy = _s06_oneside_D(ny, hy, 'B')
    else:
        Dx = _s06_oneside_D(nx, hx, 'F')
        Dy = _s06_oneside_D(ny, hy, 'F')
    return 2.0 * Sxy * np.kron(Dx, Dy)


def _s06_datum(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx, cx = _s06_grid_centers(params['xL'], params['xR'], nx)
    hy, cy = _s06_grid_centers(params['yL'], params['yR'], ny)
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    q = np.exp(-((X - params['x0']) ** 2 / (2 * params['wx'] ** 2)
                 + (Y - params['y0']) ** 2 / (2 * params['wy'] ** 2)))
    q = (1.0 - params['fsp']) * q / (q.sum() * hx * hy)
    q[int(params['isp']) - 1, int(params['jsp']) - 1] += params['fsp'] / (hx * hy)
    return q.reshape(-1)


_s06_KEYS_2D = ('xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
                'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho',
                'x0', 'y0', 'wx', 'wy', 'isp', 'jsp', 'fsp')


def _s06_validate2d(params, extra=()):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in _s06_KEYS_2D + tuple(extra):
        if k not in params:
            raise ValueError("missing key: " + k)
        if not np.isfinite(params[k]):
            raise ValueError("non-finite value for " + k)
    for nkey in ('nx', 'ny'):
        if params[nkey] < 3 or int(params[nkey]) != params[nkey]:
            raise ValueError(nkey + " must be an integer >= 3")
    if params['xR'] <= params['xL'] or params['yR'] <= params['yL']:
        raise ValueError("domain bounds out of order")
    if params['Sxx'] <= 0 or params['Syy'] <= 0 or abs(params['rho']) >= 1:
        raise ValueError("need Sxx,Syy>0 and |rho|<1")
    if not (0 <= params['fsp'] < 1):
        raise ValueError("spike fraction must lie in [0,1)")
    if not (1 <= params['isp'] <= params['nx'] and 1 <= params['jsp'] <= params['ny']):
        raise ValueError("spike cell out of range")
    if params['wx'] <= 0 or params['wy'] <= 0:
        raise ValueError("datum widths must be positive")


def _oracle_mixed_action_at_node(params):
    _s06_validate2d(params, extra=('iq', 'jq'))
    nx, ny = int(params['nx']), int(params['ny'])
    iq, jq = int(params['iq']), int(params['jq'])
    if not (1 <= iq <= nx and 1 <= jq <= ny):
        raise ValueError("query node out of range")
    A = _s06_mixed(params)
    if np.abs(A.sum(axis=0)).max() > 1e-10 * max(np.abs(A).max(), 1.0):
        raise ValueError("mixed operator lost the conservative closure")
    p0 = _s06_datum(params)
    v = A @ p0
    return float(v[(iq - 1) * ny + (jq - 1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = dict({'xL': -2.4, 'xR': 2.4, 'nx': 40, 'kx': 4.2, 'mx': 0.0, 'Sxx': 1.0, 'yL': -1.8, 'yR': 1.8, 'ny': 10, 'ky': 3.1, 'my': 0.0, 'Syy': 0.25, 'rho': 0.4, 'x0': 0.3, 'y0': -0.25, 'wx': 0.45, 'wy': 0.5, 'isp': 21, 'jsp': 6, 'fsp': 0.35, 'dt': 0.05, 'nt': 6}, iq=21, jq=6)
    neg = dict(base, rho=-0.4)
    zer = dict(base, rho=0.0)
    return [
        {"setup": "params = " + repr(base),
         "call": "mixed_action_at_node(params)",
         "gold_call": "_oracle_mixed_action_at_node(params)"},
        {"setup": "params = " + repr(neg),
         "call": "mixed_action_at_node(params)",
         "gold_call": "_oracle_mixed_action_at_node(params)"},
        {"setup": "params = " + repr(zer),
         "call": "mixed_action_at_node(params)",
         "gold_call": "_oracle_mixed_action_at_node(params)"},
    ]
