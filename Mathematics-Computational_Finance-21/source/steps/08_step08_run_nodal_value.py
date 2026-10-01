"""
Build the pinned initial datum from the configuration, advance it by nt composite steps by repeatedly calling the step 7 oracle *oracle*composite_step_density, and return the density at the report cell (istar, jstar) at final time. nt = 0 is valid and returns the initial value at that cell. Excludes the running-minimum diagnostic, which belongs to the orchestrator.

The final-time field mixes the relaxed smooth mass with the decayed spike transient, so single-node values discriminate between time integrators of equal formal order. Benchmark values for this step: with nt = 3 (otherwise frozen configuration) the value at cell (20, 7) is 0.199966185375; on the frozen run (nt = 6) the value at cell (24, 4) is 0.173573703512; with nt = 0 the value at cell (20, 7) is the datum value 0.0959029015076.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_nodal_value(params: dict) -> float:
    """Density at one report cell after nt complete composite steps.

    The nt steps must be produced by repeated calls to the step 7
    oracle _oracle_composite_step_density; this step contributes the
    pinned initial datum and the report-cell read-out.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys as in step 7, plus 'nt' (number of
        composite steps, integer >= 0) and the report cell 'istar',
        'jstar' (1-based).

    Returns
    -------
    float
        p[(istar-1)*ny + (jstar-1)] after nt steps (0-based storage,
        x-major ordering).

    Raises
    ------
    ValueError
        On invalid inputs, dt <= 0, nt < 0 or non-integer, or a report
        cell out of range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s08_grid_centers(aL, aR, n):
    h = (aR - aL) / n
    return h, aL + (np.arange(1, n + 1) - 0.5) * h


def _s08_datum(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx, cx = _s08_grid_centers(params['xL'], params['xR'], nx)
    hy, cy = _s08_grid_centers(params['yL'], params['yR'], ny)
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    q = np.exp(-((X - params['x0']) ** 2 / (2 * params['wx'] ** 2)
                 + (Y - params['y0']) ** 2 / (2 * params['wy'] ** 2)))
    q = (1.0 - params['fsp']) * q / (q.sum() * hx * hy)
    q[int(params['isp']) - 1, int(params['jsp']) - 1] += params['fsp'] / (hx * hy)
    return q.reshape(-1)


_s08_KEYS_2D = ('xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
                'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho',
                'x0', 'y0', 'wx', 'wy', 'isp', 'jsp', 'fsp')


def _s08_validate2d(params, extra=()):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in _s08_KEYS_2D + tuple(extra):
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

def _oracle_run_nodal_value(params):
    _s08_validate2d(params, extra=('dt', 'nt', 'istar', 'jstar'))
    if params['dt'] <= 0:
        raise ValueError("dt must be positive")
    nt = params['nt']
    if nt < 0 or int(nt) != nt:
        raise ValueError("nt must be a nonnegative integer")
    nx, ny = int(params['nx']), int(params['ny'])
    if not (1 <= params['istar'] <= nx and 1 <= params['jstar'] <= ny):
        raise ValueError("report cell out of range")
    p = _s08_datum(params)
    for _ in range(int(nt)):
        p = _oracle_composite_step_density(params, p)
    ell = (int(params['istar']) - 1) * ny + (int(params['jstar']) - 1)
    return float(p[ell])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = dict({'xL': -2.4, 'xR': 2.4, 'nx': 40, 'kx': 4.2, 'mx': 0.0, 'Sxx': 1.0, 'yL': -1.8, 'yR': 1.8, 'ny': 10, 'ky': 3.1, 'my': 0.0, 'Syy': 0.25, 'rho': 0.4, 'x0': 0.3, 'y0': -0.25, 'wx': 0.45, 'wy': 0.5, 'isp': 21, 'jsp': 6, 'fsp': 0.35, 'dt': 0.05, 'nt': 6}, istar=20, jstar=7)
    c1 = dict(base, nt=3)
    c2 = dict(base, istar=24, jstar=4)
    c3 = dict(base, nt=0)
    return [
        {"setup": "params = " + repr(c1),
         "call": "run_nodal_value(params)",
         "gold_call": "_oracle_run_nodal_value(params)"},
        {"setup": "params = " + repr(c2),
         "call": "run_nodal_value(params)",
         "gold_call": "_oracle_run_nodal_value(params)"},
        {"setup": "params = " + repr(c3),
         "call": "run_nodal_value(params)",
         "gold_call": "_oracle_run_nodal_value(params)"},
    ]
