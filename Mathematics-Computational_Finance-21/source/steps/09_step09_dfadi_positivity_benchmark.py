"""
Orchestrator: run the complete positivity benchmark and return the most negative nodal value attained over the nt complete composite steps. The run itself is built from the earlier steps: the pinned datum is advanced by repeated calls to the step 7 oracle, and the minimum is taken over the states those calls return. Calls the oracles of steps 1-8 and uses every output: EM structure (step 1) and Perron positivity (step 2) as admissibility checks, the four thresholds (steps 3-4) as finiteness/positivity checks, the directional-factor action (step 5) and mixed action (step 6) as consistency checks against its own assembled factors, the step 7 oracle as the engine that produces every state in the run, and the step 8 oracle as a cross-check on the final-time report cell; it finally verifies discrete mass conservation.

For the frozen configuration the per-step minima after steps 1..6 are -0.125037782569, -0.0801128830357, -0.0545215832420, -0.0390413567081, -0.0277721597472, -0.0209386688392, so the most negative value over the run is attained at the first step, during the stiff response of the sub-threshold directional factors to the spike: the final benchmark scalar is -0.125037782569066. Variant anchors: dt = 0.04 with nt = 8 gives -0.140951045948437 and fsp = 0.2 gives -0.0392243030528.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dfadi_positivity_benchmark(params: dict) -> float:
    """Most negative nodal value over the full composite-step run.

    Parameters
    ----------
    params : dict
        The complete frozen-configuration dictionary: the 2-D keys of
        step 6, the run keys 'dt', 'nt' (nt >= 1), the report cell
        'istar', 'jstar', and the threshold keys 'glo', 'ghi',
        'tol_pos', 'nbis'.

    Returns
    -------
    float
        min over n = 1..nt of the minimum entry of the density after
        complete composite step n.

    Raises
    ------
    ValueError
        On invalid inputs, or when any cross-check against the earlier
        steps fails (EM structure, stationary positivity, threshold
        finiteness, factor consistency, report-cell cross-check, mass
        conservation).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_dfadi_positivity_benchmark(params):
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
    if _oracle_min_offdiagonal_entry(px) >= 0:
        raise ValueError("x generator is not EM proper for this configuration")
    for pdir in (px, py):
        if _oracle_stationary_minimum_mass(pdir) <= 0:
            raise ValueError("stationary density is not strictly positive")

    # steps 3-4: thresholds exist, are finite, and the Pade ones positive
    g0 = [_oracle_first_order_threshold(dict(pdir, **thr)) for pdir in (px, py)]
    gr = [_oracle_second_order_threshold(dict(pdir, **thr)) for pdir in (px, py)]
    if not all(np.isfinite(g0)) or not all(np.isfinite(gr)):
        raise ValueError("threshold computation failed")
    if min(gr) <= 0:
        raise ValueError("expected strictly positive Pade thresholds")

    # step 5: delta action of the x factor at the spike cell
    sub = params['dt'] / 2.0
    q_min = _oracle_directional_factor_delta_action(dict(px, s=sub, k=params['isp']))
    Lx, Ly, Axy = _s09_operators(params)
    Mx = np.eye(nx) - sub * Lx + (sub * sub / 2.0) * (Lx @ Lx)
    e = np.zeros(nx)
    e[int(params['isp']) - 1] = 1.0
    if abs(float(np.linalg.solve(Mx, e).min()) - q_min) > 1e-11:
        raise ValueError("directional factor inconsistent with step 5")

    # step 6: mixed action at the report node
    v_star = _oracle_mixed_action_at_node(dict(params, iq=params['istar'],
                                               jq=params['jstar']))
    p0 = _s09_datum(params)
    ell = (int(params['istar']) - 1) * ny + (int(params['jstar']) - 1)
    if abs(float((Axy @ p0)[ell]) - v_star) > 1e-9 * max(1.0, abs(v_star)):
        raise ValueError("mixed operator inconsistent with step 6")

    # the run: every state produced by the step 7 oracle
    p = p0.copy()
    minrun = np.inf
    for _ in range(nt):
        p = _oracle_composite_step_density(params, p)
        minrun = min(minrun, float(p.min()))

    # step 8: cross-check the final-time report cell
    s8 = _oracle_run_nodal_value(params)
    if abs(float(p[ell]) - s8) > 1e-11:
        raise ValueError("final nodal value inconsistent with step 8")

    mass = float(p.sum()) * hx * hy
    if abs(mass - 1.0) > 1e-10:
        raise ValueError("discrete mass is not conserved")
    return float(minrun)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    frozen = dict({'xL': -2.4, 'xR': 2.4, 'nx': 40, 'kx': 4.2, 'mx': 0.0, 'Sxx': 1.0, 'yL': -1.8, 'yR': 1.8, 'ny': 10, 'ky': 3.1, 'my': 0.0, 'Syy': 0.25, 'rho': 0.4, 'x0': 0.3, 'y0': -0.25, 'wx': 0.45, 'wy': 0.5, 'isp': 21, 'jsp': 6, 'fsp': 0.35, 'dt': 0.05, 'nt': 6}, istar=20, jstar=7,
                  glo=1e-6, ghi=64.0, tol_pos=-1e-12, nbis=200)
    v1 = dict(frozen, dt=0.04, nt=8)
    v2 = dict(frozen, fsp=0.2)
    return [
        {"setup": "params = " + repr(frozen),
         "call": "dfadi_positivity_benchmark(params)",
         "gold_call": "_oracle_dfadi_positivity_benchmark(params)"},
        {"setup": "params = " + repr(v1),
         "call": "dfadi_positivity_benchmark(params)",
         "gold_call": "_oracle_dfadi_positivity_benchmark(params)"},
        {"setup": "params = " + repr(v2),
         "call": "dfadi_positivity_benchmark(params)",
         "gold_call": "_oracle_dfadi_positivity_benchmark(params)"},
    ]
