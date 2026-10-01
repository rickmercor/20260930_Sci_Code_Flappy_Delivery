#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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


def min_offdiagonal_entry(params):
    _s01_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma'))
    L, _ = _s01_operator(params['aL'], params['aR'], int(params['n']),
                         params['kappa'], params['m'], params['sigma'])
    off = L.copy()
    np.fill_diagonal(off, 0.0)
    if np.abs(L.sum(axis=0)).max() > 1e-10 * np.abs(L).max():
        raise ValueError("conservative closure violated")
    return float(off.min())

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


def stationary_minimum_mass(params):
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


def first_order_threshold(params):
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

import numpy as np


def _s04_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s04_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s04_grid(aL, aR, n)
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


def _s04_validate1d(params, keys):
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

def _s04_bisect(minfun, glo, ghi, tol_pos, nbis):
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


def second_order_threshold(params):
    _s04_validate1d(params, ('aL', 'aR', 'n', 'kappa', 'm', 'sigma',
                             'glo', 'ghi', 'tol_pos', 'nbis'))
    if not (0 < params['glo'] < params['ghi']):
        raise ValueError("need 0 < glo < ghi")
    n = int(params['n'])
    L, _ = _s04_operator(params['aL'], params['aR'], n,
                         params['kappa'], params['m'], params['sigma'])
    I = np.eye(n)
    L2 = L @ L

    def minfun(g):
        return np.linalg.inv(I - g * L + (g * g / 2.0) * L2).min()

    return _s04_bisect(minfun, params['glo'], params['ghi'],
                       params['tol_pos'], params['nbis'])

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


def directional_factor_delta_action(params):
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


def mixed_action_at_node(params):
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


def composite_step_density(params, p):
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

def run_nodal_value(params):
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
        p = composite_step_density(params, p)
    ell = (int(params['istar']) - 1) * ny + (int(params['jstar']) - 1)
    return float(p[ell])

import numpy as np


def _s09_grid(aL, aR, n):
    h = (aR - aL) / n
    faces = aL + np.arange(1, n) * h
    return h, faces


def _s09_operator(aL, aR, n, kappa, m, sigma):
    """Flux-form 1-D Fokker-Planck generator, zero-flux closure, second-order
    linear-upwind convective face value with first-order upwind fallback at
    the boundary, central diffusive flux."""
    h, faces = _s09_grid(aL, aR, n)
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


def _s09_validate1d(params, keys):
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

def _s09_grid_centers(aL, aR, n):
    h = (aR - aL) / n
    return h, aL + (np.arange(1, n + 1) - 0.5) * h


def _s09_oneside_D(n, h, side):
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


def _s09_mixed(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx = (params['xR'] - params['xL']) / nx
    hy = (params['yR'] - params['yL']) / ny
    Sxy = params['rho'] * np.sqrt(params['Sxx'] * params['Syy'])
    if Sxy >= 0:
        Dx = _s09_oneside_D(nx, hx, 'F')
        Dy = _s09_oneside_D(ny, hy, 'B')
    else:
        Dx = _s09_oneside_D(nx, hx, 'F')
        Dy = _s09_oneside_D(ny, hy, 'F')
    return 2.0 * Sxy * np.kron(Dx, Dy)


def _s09_datum(params):
    nx, ny = int(params['nx']), int(params['ny'])
    hx, cx = _s09_grid_centers(params['xL'], params['xR'], nx)
    hy, cy = _s09_grid_centers(params['yL'], params['yR'], ny)
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    q = np.exp(-((X - params['x0']) ** 2 / (2 * params['wx'] ** 2)
                 + (Y - params['y0']) ** 2 / (2 * params['wy'] ** 2)))
    q = (1.0 - params['fsp']) * q / (q.sum() * hx * hy)
    q[int(params['isp']) - 1, int(params['jsp']) - 1] += params['fsp'] / (hx * hy)
    return q.reshape(-1)


_s09_KEYS_2D = ('xL', 'xR', 'nx', 'kx', 'mx', 'Sxx',
                'yL', 'yR', 'ny', 'ky', 'my', 'Syy', 'rho',
                'x0', 'y0', 'wx', 'wy', 'isp', 'jsp', 'fsp')


def _s09_validate2d(params, extra=()):
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for k in _s09_KEYS_2D + tuple(extra):
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

def _s09_pade_matrix(L, s):
    n = L.shape[0]
    return np.eye(n) - s * L + (s * s / 2.0) * (L @ L)


def _s09_operators(params):
    Lx, _ = _s09_operator(params['xL'], params['xR'], int(params['nx']),
                          params['kx'], params['mx'], params['Sxx'])
    Ly, _ = _s09_operator(params['yL'], params['yR'], int(params['ny']),
                          params['ky'], params['my'], params['Syy'])
    return Lx, Ly, _s09_mixed(params)


def dfadi_positivity_benchmark(params):
    _s09_validate2d(params, extra=('dt', 'nt', 'istar', 'jstar',
                                   'glo', 'ghi', 'tol_pos', 'nbis'))
    nt = int(params['nt'])
    if nt < 1:
        raise ValueError("the benchmark run needs nt >= 1")
    nx, ny = int(params['nx']), int(params['ny'])
    hx = (params['xR'] - params['xL']) / nx
    hy = (params['yR'] - params['yL']) / ny

    px = dict(aL=params['xL'], aR=params['xR'], n=params['nx'],
              kappa=params['kx'], m=params['mx'], sigma=params['Sxx'])
    py = dict(aL=params['yL'], aR=params['yR'], n=params['ny'],
              kappa=params['ky'], m=params['my'], sigma=params['Syy'])
    thr = dict(glo=params['glo'], ghi=params['ghi'],
               tol_pos=params['tol_pos'], nbis=params['nbis'])

    # steps 1-2: EM structure and Perron positivity
    if min_offdiagonal_entry(px) >= 0:
        raise ValueError("x generator is not EM proper for this configuration")
    for pdir in (px, py):
        if stationary_minimum_mass(pdir) <= 0:
            raise ValueError("stationary density is not strictly positive")

    # steps 3-4: thresholds exist, are finite, and the Pade ones positive
    g0 = [first_order_threshold(dict(pdir, **thr)) for pdir in (px, py)]
    gr = [second_order_threshold(dict(pdir, **thr)) for pdir in (px, py)]
    if not all(np.isfinite(g0)) or not all(np.isfinite(gr)):
        raise ValueError("threshold computation failed")
    if min(gr) <= 0:
        raise ValueError("expected strictly positive Pade thresholds")

    # step 5: delta action of the x factor at the spike cell
    sub = params['dt'] / 2.0
    q_min = directional_factor_delta_action(dict(px, s=sub, k=params['isp']))
    Lx, Ly, Axy = _s09_operators(params)
    Mx = np.eye(nx) - sub * Lx + (sub * sub / 2.0) * (Lx @ Lx)
    e = np.zeros(nx)
    e[int(params['isp']) - 1] = 1.0
    if abs(float(np.linalg.solve(Mx, e).min()) - q_min) > 1e-11:
        raise ValueError("directional factor inconsistent with step 5")

    # step 6: mixed action at the report node
    v_star = mixed_action_at_node(dict(params, iq=params['istar'],
                                               jq=params['jstar']))
    p0 = _s09_datum(params)
    ell = (int(params['istar']) - 1) * ny + (int(params['jstar']) - 1)
    if abs(float((Axy @ p0)[ell]) - v_star) > 1e-9 * max(1.0, abs(v_star)):
        raise ValueError("mixed operator inconsistent with step 6")

    # the run: every state produced by the step 7 oracle
    p = p0.copy()
    minrun = np.inf
    for _ in range(nt):
        p = composite_step_density(params, p)
        minrun = min(minrun, float(p.min()))

    # step 8: cross-check the final-time report cell
    s8 = run_nodal_value(params)
    if abs(float(p[ell]) - s8) > 1e-11:
        raise ValueError("final nodal value inconsistent with step 8")

    mass = float(p.sum()) * hx * hy
    if abs(mass - 1.0) > 1e-10:
        raise ValueError("discrete mass is not conserved")
    return float(minrun)
SCICODE_GOLD_EOF
