"""
Advance a given density vector by exactly one complete composite step of the source's arrangement - the two directional factors and the mixed-derivative factor, each applied by a direct dense solve at the substep length the source assigns it - and return the new density vector. The arrangement, the directional map and the mixed-block factor are withheld ingredients of the problem statement and are not restated here; the step must conserve discrete mass exactly and be second-order accurate in dt. This is the single reusable building block of the benchmark: the runs of steps 8 and 9 are built by repeated calls to this step.

One composite step reads p1 = Fx(dt/2) Fy(dt/2) Fxy(dt) Fy(dt/2) Fx(dt/2) p0 with Fa(s) q = (I - s La + s^2 La^2/2)^(-1) q applied line by line and Fxy(dt) = (I - dt/2 Axy)^(-1)(I + dt/2 Axy). Benchmark values for this step: at dt = 0.04 on the otherwise frozen configuration the one-step minimum is -0.140951045948; with the spike removed (fsp = 0, dt = 0.05) it is -3.63134236814e-04; with the cross term removed (rho = 0, dt = 0.05) it is +4.52599821264e-10, entrywise positive to that level because the composite degenerates to the separable directional product.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def composite_step_density(params: dict, p: np.ndarray) -> np.ndarray:
    """Density after one complete composite step applied to a given state.

    Parameters
    ----------
    params : dict
        The 2-D configuration keys as in step 6 (without 'iq'/'jq'),
        plus 'dt' (> 0). One call advances the state by one complete
        composite step of the source's arrangement: every factor - the
        two directional ones and the mixed-derivative one - is applied
        by a direct dense solve, at the substep length the source
        assigns that factor. The arrangement itself, the directional map
        and the mixed-block factor are withheld ingredients of the
        problem statement and are not restated here.
    p : numpy.ndarray
        Density vector of length nx*ny (0-based, x-major ordering) to
        advance; it is not modified.

    Returns
    -------
    numpy.ndarray
        The density vector after the step, with the same length as p.

    Raises
    ------
    ValueError
        On invalid inputs, dt <= 0, or p of the wrong length or with
        non-finite entries.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s07_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s07_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s07_grid(aL, aR, n)
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


def _s07_validate1d(params, keys):
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

def _s07_grid_centers(aL, aR, n):
    h = (aR - aL) / n
    return h, aL + (np.arange(1, n + 1) - 0.5) * h


def _s07_oneside_D(n, h, side):
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


def _s07_mixed(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx = (params['xR'] - params['xL']) / nx
    hy = (params['yR'] - params['yL']) / ny
    Sxy = params['rho'] * np.sqrt(params['Sxx'] * params['Syy'])
    if Sxy >= 0:
        Dx = _s07_oneside_D(nx, hx, 'F')
        Dy = _s07_oneside_D(ny, hy, 'B')
    else:
        Dx = _s07_oneside_D(nx, hx, 'F')
        Dy = _s07_oneside_D(ny, hy, 'F')
    return 2.0 * Sxy * np.kron(Dx, Dy)


def _s07_datum(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx, cx = _s07_grid_centers(params['xL'], params['xR'], nx)
    hy, cy = _s07_grid_centers(params['yL'], params['yR'], ny)
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    q = np.exp(-((X - params['x0']) ** 2 / (2 * params['wx'] ** 2)
                 + (Y - params['y0']) ** 2 / (2 * params['wy'] ** 2)))
    q = (1.0 - params['fsp']) * q / (q.sum() * hx * hy)
    q[int(params['isp']) - 1, int(params['jsp']) - 1] += params['fsp'] / (hx * hy)
    return q.reshape(-1)


_s07_KEYS_2D = ('xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
                'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho',
                'x0', 'y0', 'wx', 'wy', 'isp', 'jsp', 'fsp')


def _s07_validate2d(params, extra=()):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in _s07_KEYS_2D + tuple(extra):
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

def _s07_pade_matrix(L, s):
    n = L.shape[0]
    return np.eye(n) - s * L + (s * s / 2.0) * (L @ L)


def _s07_composite_step(p, Lx, Ly, Axy, dt, nx, ny):
    """One mirrored composite step: x and y directional Pade(0,2) factors at
    dt/2, the trapezoidal mixed factor at dt in the centre, then y and x
    again, every factor applied by a direct dense solve."""
    s = dt / 2.0
    Mx = _s07_pade_matrix(Lx, s)
    My = _s07_pade_matrix(Ly, s)
    N = nx * ny
    Mc = np.eye(N) - (dt / 2.0) * Axy
    P = p.reshape(nx, ny)
    P = np.linalg.solve(Mx, P)
    P = np.linalg.solve(My, P.T).T
    q = P.reshape(-1)
    q = np.linalg.solve(Mc, q + (dt / 2.0) * (Axy @ q))
    P = q.reshape(nx, ny)
    P = np.linalg.solve(My, P.T).T
    P = np.linalg.solve(Mx, P)
    return P.reshape(-1)


def _s07_operators(params):
    Lx, _ = _s07_operator(params['xL'], params['xR'], int(params['nx']),
                          params['kx'], params['mx'], params['Sxx'])
    Ly, _ = _s07_operator(params['yL'], params['yR'], int(params['ny']),
                          params['ky'], params['my'], params['Syy'])
    return Lx, Ly, _s07_mixed(params)


def _oracle_composite_step_density(params, p):
    _s07_validate2d(params, extra=('dt',))
    if params['dt'] <= 0:
        raise ValueError("dt must be positive")
    nx, ny = int(params['nx']), int(params['ny'])
    q = np.asarray(p, dtype=float).reshape(-1)
    if q.size != nx * ny:
        raise ValueError("p must have length nx*ny")
    if not np.all(np.isfinite(q)):
        raise ValueError("p must be finite")
    Lx, Ly, Axy = _s07_operators(params)
    return _s07_composite_step(q, Lx, Ly, Axy, params['dt'], nx, ny)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = dict({'xL': -2.4, 'xR': 2.4, 'nx': 40, 'kx': 4.2, 'mx': 0.0, 'Sxx': 1.0, 'yL': -1.8, 'yR': 1.8, 'ny': 10, 'ky': 3.1, 'my': 0.0, 'Syy': 0.25, 'rho': 0.4, 'x0': 0.3, 'y0': -0.25, 'wx': 0.45, 'wy': 0.5, 'isp': 21, 'jsp': 6, 'fsp': 0.35, 'dt': 0.05, 'nt': 6})
    v1 = dict(base, fsp=0.2)
    v2 = dict(base, dt=0.04)
    v3 = dict(base, fsp=0.0)
    datum = "; import numpy as np; nx=int(params['nx']); ny=int(params['ny']); hx=(params['xR']-params['xL'])/nx; hy=(params['yR']-params['yL'])/ny; cx=params['xL']+(np.arange(1,nx+1)-0.5)*hx; cy=params['yL']+(np.arange(1,ny+1)-0.5)*hy; X,Y=np.meshgrid(cx,cy,indexing='ij'); q=np.exp(-((X-params['x0'])**2/(2*params['wx']**2)+(Y-params['y0'])**2/(2*params['wy']**2))); q=(1.0-params['fsp'])*q/(q.sum()*hx*hy); q[int(params['isp'])-1,int(params['jsp'])-1]+=params['fsp']/(hx*hy); p=q.reshape(-1)"
    return [
        {"setup": "params = " + repr(base) + datum,
         "call": "float(composite_step_density(params, p).min())",
         "gold_call": "float(_oracle_composite_step_density(params, p).min())"},
        {"setup": "params = " + repr(v1) + datum,
         "call": "float(composite_step_density(params, p)[196])",
         "gold_call": "float(_oracle_composite_step_density(params, p)[196])"},
        {"setup": "params = " + repr(v2) + datum,
         "call": "float(composite_step_density(params, p).max())",
         "gold_call": "float(_oracle_composite_step_density(params, p).max())"},
        {"setup": "params = " + repr(v3) + datum,
         "call": "float(composite_step_density(params, p)[0])",
         "gold_call": "float(_oracle_composite_step_density(params, p)[0])"},
    ]
