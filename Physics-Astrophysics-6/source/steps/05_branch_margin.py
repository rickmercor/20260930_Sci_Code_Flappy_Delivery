"""
Transverse strength of the admissible field above one periodic plane.

Transverse strength of the admissible field above one periodic plane.

The admissible field has unit magnitude, vanishing divergence, positive y component and
x,y periodicity, carries the prescribed longitudinal component everywhere and the
prescribed normal component on z=0. Above the plane x=x0 its y component varies with
height; this step reports how small that component becomes below a queried height.

Returns
-------
The minimum y component of the admissible field above the plane x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def branch_margin(x, zmax, phases, bounds):
    """Return the minimum y component of the admissible field above the plane x.

    x is a finite real scalar and zmax a finite real height with zmax>=0. phases and
    bounds follow the seed_bounds conventions. The minimum runs over the whole periodic
    y line and over 0<=z<=zmax, and is returned as one finite float.

    Raises
    ------
    ValueError
        If any input is malformed or nonfinite, if zmax is negative, or if the stated
        properties cannot be maintained above this plane throughout 0<=z<=zmax.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

_TWO_PI = 2 * np.pi


def _two_scalars(first, second):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.ndim or b.ndim or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError('both arguments must be finite real scalars')
    if b < 0:
        raise ValueError('the height argument must not be negative')
    return float(a), float(b)


def _theta_pack(phases, bounds, xpk, y, z):
    """theta above the plane, with the derivatives the continuation needs."""
    sx, dsx = xpk
    sy = _wave(y, phases[1])
    dsy = _wave(y, phases[1], 1)
    d2sy = _wave(y, phases[1], 2)
    sz = _wave(np.asarray(z, dtype=float), phases[2])
    dsz = _wave(np.asarray(z, dtype=float), phases[2], 1)
    span = np.diff(bounds[1])[0]
    beta = (np.pi - .2) / np.expm1(5.)
    u = (sx * sy * sz - bounds[1, 0]) / span
    theta = .1 + beta * np.expm1(5 * u)
    fp = 5 * beta * np.exp(5 * u)
    fpp = 5 * fp
    ux, uy, uz = dsx * sy * sz / span, sx * dsy * sz / span, sx * sy * dsz / span
    uxy, uyy, uzy = dsx * dsy * sz / span, sx * d2sy * sz / span, sx * dsy * dsz / span
    return (theta, fp * ux, fp * uy, fp * uz,
            fpp * ux * uy + fp * uxy, fpp * uy * uy + fp * uyy, fpp * uz * uy + fp * uzy)


def _azimuth0(phases, bounds, x, y):
    v = (_wave(x, phases[0]) * _wave(y, phases[1]) - bounds[0, 0]) / np.diff(bounds[0])[0]
    return 1 + (np.pi - 2) * v


def _dazimuth0(phases, bounds, x, y):
    return (np.pi - 2) * _wave(x, phases[0]) * _wave(y, phases[1], 1) / np.diff(bounds[0])[0]


def _transport(phases, bounds, xpk, z, y, ph, jac, kap):
    """Right-hand side of the transported state (y, azimuth, and their y0 sensitivities)."""
    th, tx, ty, tz, txy, tyy, tzy = _theta_pack(phases, bounds, xpk, y, z)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)
    cott, dcott, cotp = ct / st, -1. / st ** 2, cp / sp
    forcing = -tx / sp + cott * ty + cott * cotp * tz
    force_y = -txy / sp + dcott * ty * ty + cott * tyy + (dcott * ty * tz + cott * tzy) * cotp
    force_p = tx * cp / sp ** 2 - cott * tz / sp ** 2
    return -cotp, forcing, kap / sp ** 2, force_y * jac + force_p * kap


def _march(phases, bounds, xs, y0s, dz, zcap, track_margin=False):
    """Vectorised fixed-step march of the transported state over a footpoint family."""
    grid_x = np.repeat(np.asarray(xs, dtype=float)[:, None], len(y0s), 1)
    grid_y = np.repeat(np.asarray(y0s, dtype=float)[None, :], len(xs), 0)
    xpk = (_wave(grid_x, phases[0]), _wave(grid_x, phases[0], 1))
    y, ph = grid_y.copy(), _azimuth0(phases, bounds, grid_x, grid_y)
    jac, kap = np.ones_like(y), _dazimuth0(phases, bounds, grid_x, grid_y)
    stop = np.full(y.shape, np.inf)
    margin = np.sin(_theta_pack(phases, bounds, xpk, y, 0.)[0]) * np.sin(ph) if track_margin else None
    z = 0.
    with np.errstate(all='ignore'):
        while z < zcap - 1e-15:
            step = min(dz, zcap - z)
            a1, b1, c1, d1 = _transport(phases, bounds, xpk, z, y, ph, jac, kap)
            a2, b2, c2, d2 = _transport(phases, bounds, xpk, z + step / 2, y + step / 2 * a1,
                                        ph + step / 2 * b1, jac + step / 2 * c1, kap + step / 2 * d1)
            a3, b3, c3, d3 = _transport(phases, bounds, xpk, z + step / 2, y + step / 2 * a2,
                                        ph + step / 2 * b2, jac + step / 2 * c2, kap + step / 2 * d2)
            a4, b4, c4, d4 = _transport(phases, bounds, xpk, z + step, y + step * a3,
                                        ph + step * b3, jac + step * c3, kap + step * d3)
            yn = y + step / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            pn = ph + step / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            jn = jac + step / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
            kn = kap + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
            folded = (jac > 0) & (jn <= 0) & np.isinf(stop)
            stop[folded] = z + step * jac[folded] / (jac[folded] - jn[folded])
            sp, spn = np.sin(ph), np.sin(pn)
            lost = (sp > 0) & (spn <= 0) & np.isinf(stop)
            stop[lost] = z + step * sp[lost] / (sp[lost] - spn[lost])
            blown = ~np.isfinite(yn + pn + jn + kn) & np.isinf(stop)
            stop[blown] = z + step
            y, ph, jac, kap = yn, pn, jn, kn
            z += step
            if track_margin:
                margin = np.fmin(margin, np.sin(_theta_pack(phases, bounds, xpk, y, z)[0]) * np.sin(ph))
            if np.isfinite(stop).any() and z > 1.3 * np.nanmin(stop[np.isfinite(stop)]):
                break
    return stop, margin


def _breakdown(phases, bounds, x, y0, ceiling):
    """Height at which the transported state above (x, y0) first stops being admissible."""
    xpk = (_wave(np.float64(x), phases[0]), _wave(np.float64(x), phases[0], 1))

    def rhs(z, s):
        return _transport(phases, bounds, xpk, z, s[0], s[1], s[2], s[3])

    def fold(z, s):
        return s[2]
    fold.terminal, fold.direction = True, -1

    def branch(z, s):
        return np.sin(s[1]) - 1e-6
    branch.terminal, branch.direction = True, -1

    state = [float(y0), float(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))), 1.,
             float(_dazimuth0(phases, bounds, np.float64(x), np.float64(y0)))]
    sol = solve_ivp(rhs, (0., float(ceiling)), state, method='DOP853', rtol=1e-11, atol=1e-13,
                    events=(fold, branch))
    hits = [float(e[0]) for e in sol.t_events if e.size]
    if hits:
        return min(hits)
    # The transport is singular where the azimuth reaches 0 or pi, so an integration that
    # cannot be continued has reached the loss of the positive branch there.
    return float(sol.t[-1]) if sol.status == -1 else np.inf


def _margin_along(phases, bounds, x, y0, zmax):
    """Minimum y component of the field along the characteristic above (x, y0) up to zmax."""
    xpk = (_wave(np.float64(x), phases[0]), _wave(np.float64(x), phases[0], 1))
    if zmax == 0.:
        th = _theta_pack(phases, bounds, xpk, np.float64(y0), 0.)[0]
        return float(np.sin(th) * np.sin(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))))

    def rhs(z, s):
        return _transport(phases, bounds, xpk, z, s[0], s[1], s[2], s[3])

    state = [float(y0), float(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))), 1.,
             float(_dazimuth0(phases, bounds, np.float64(x), np.float64(y0)))]
    sol = solve_ivp(rhs, (0., float(zmax)), state, method='DOP853', rtol=1e-12, atol=1e-14,
                    dense_output=True)
    if not sol.success:
        raise ValueError('the transported state could not be resolved')

    def profile(zs):
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        s = sol.sol(zs)
        return np.sin(_theta_pack(phases, bounds, xpk, s[0], zs)[0]) * np.sin(s[1])
    grid = np.linspace(0., float(zmax), 201)
    vals = profile(grid)
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(lambda zz: float(profile(zz)[0]), bounds=(lo, hi),
                              method='bounded', options={'xatol': 1e-12})
        return float(min(vals[k], res.fun))
    return float(vals[k])


def _oracle_branch_margin(x, zmax, phases, bounds):
    phases = _check_phases(phases)
    bounds = _check_bounds(bounds)
    x, zmax = _two_scalars(x, zmax)
    family = _TWO_PI * np.arange(256) / 256
    if zmax > 0:
        stop, margin = _march(phases, bounds, [x], family, zmax / 400., zmax, track_margin=True)
        if np.any(np.isfinite(stop)):
            raise ValueError('the properties cannot be maintained above this plane up to zmax')
        order = np.argsort(margin[0])[:3]
    else:
        th = _theta_pack(phases, bounds,
                         (_wave(np.full(len(family), x), phases[0]),
                          _wave(np.full(len(family), x), phases[0], 1)), family, 0.)[0]
        order = np.argsort(np.sin(th) * np.sin(_azimuth0(phases, bounds, x, family)))[:3]
    best = np.inf
    for j in order:
        spacing = _TWO_PI / len(family)
        fun = lambda t: _margin_along(phases, bounds, x, t, zmax)
        left, right = family[j] - spacing, family[j] + spacing
        res = minimize_scalar(fun, bounds=(left, right), method='bounded',
                              options={'xatol': 1e-9})
        best = min(best, float(res.fun), fun(family[j]))
    if not np.isfinite(best):
        raise ValueError('the field above this plane is not finite')
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n',
      'call': 'branch_margin(2.2143518802, 0.3, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_branch_margin(2.2143518802, 0.3, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n',
      'call': 'branch_margin(1.0, 0.0, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_branch_margin(1.0, 0.0, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\np = np.round(np.random.default_rng(303).uniform(0, 2 * np.pi, (3, 5)), 2)\n',
      'call': 'branch_margin(3.4335205081, 0.25, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_branch_margin(3.4335205081, 0.25, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n'
               '\n'
               'def case():\n'
               '    _case_bounds = seed_bounds(p.copy())\n'
               '    try:\n'
               '        branch_margin(1.0, -0.1, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    _case_bounds = _oracle_seed_bounds(p.copy())\n'
               '    try:\n'
               '        _oracle_branch_margin(1.0, -0.1, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n'
               '\n'
               'def case2():\n'
               '    _case_bounds = seed_bounds(p.copy())\n'
               '    try:\n'
               '        branch_margin(2.2143518802, 1.5, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case2():\n'
               '    _case_bounds = _oracle_seed_bounds(p.copy())\n'
               '    try:\n'
               '        _oracle_branch_margin(2.2143518802, 1.5, p.copy(), _case_bounds)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case2()',
      'gold_call': 'gold_case2()'}]
