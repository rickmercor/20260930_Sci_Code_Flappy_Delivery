"""
Final orchestrator: the height to which the admissible field exists.

Final orchestrator: the height to which the admissible field exists.

Combine the preceding public functions, without redefining their scientific operations,
to return the requested scalar for the prescribed instance: the supremum of heights for
which a continuously differentiable field with every stated property exists over the
whole periodic domain.

Returns
-------
The supremum height reached by the admissible field, as one float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def existence_height(phases=None, ceiling=2.0):
    """Return the supremum height reached by the admissible field, as one float.

    phases is a finite (3,m) array under the seed_bounds conventions; when omitted its
    rows are (5.93,2.26,4.93,3.72,1.85), (5.80,5.46,2.29,6.11,1.41) and
    (5.06,4.28,2.96,0.19,5.62). ceiling is a finite positive search limit. Use the
    earlier functions for the corresponding operations.

    Raises
    ------
    ValueError
        If phases or ceiling are malformed or nonfinite, if the field satisfies every
        stated property throughout the search range, or if the reconstructions of the
        field disagree with one another or with the prescribed data.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_TWO_PI = 2 * np.pi


def _global_candidates(phases, bounds, ceiling, nx=24, ny0=48, keep=4, reach=1.3, most=8):
    """Coarse locator: starting cells in every basin a two-dimensional family can see.

    The earliest-stopping cells, plus the earliest cell of every plane that is a local
    minimum along x, limited to cells stopping within `reach` of the earliest one. One
    earliest cell is not enough: separate basins can sit within a few percent of each
    other, and a coarse family can rank them in the wrong order.
    """
    xs = _TWO_PI * np.arange(nx) / nx
    y0s = _TWO_PI * np.arange(ny0) / ny0
    stop, _ = _march(phases, bounds, xs, y0s, min(ceiling, 1.0) / 100., ceiling)
    if not np.isfinite(stop).any():
        raise ValueError('no breakdown of the field was located below the search range')
    flat = np.where(np.isfinite(stop), stop, np.inf)
    picks = [np.unravel_index(int(k), flat.shape)
             for k in np.argsort(flat, axis=None, kind='stable')[:keep]]
    column = flat.min(axis=1)
    for i in np.argsort(column, kind='stable'):
        if np.isfinite(column[i]) and column[i] <= column[i - 1] and column[i] <= column[(i + 1) % nx]:
            picks.append((int(i), int(np.argmin(flat[i]))))
    limit = reach * flat.min()
    starts = []
    for i, j in picks:
        cell = (float(xs[i]), float(y0s[j]))
        if flat[i, j] <= limit and cell not in starts:
            starts.append(cell)
    return starts[:most], _TWO_PI / nx, _TWO_PI / ny0


def _refine_pair(phases, bounds, x0, y0, hx, hy, ceiling, rough=False):
    """Local grid then Nelder-Mead on the two-variable breakdown height.

    A rough pass ranks the basins cheaply; the final pass resolves the winner. The final
    tolerances sit just above the event-location noise of the integrator (about 1e-14 in
    height), because a tighter function tolerance is never met and only exhausts the
    evaluation budget.
    """
    grid_x = x0 + hx * np.linspace(-.5, .5, 5)
    grid_y = y0 + hy * np.linspace(-.5, .5, 5)
    best = (np.inf, x0, y0)
    for xv in grid_x:
        for yv in grid_y:
            value = _breakdown(phases, bounds, xv, yv, ceiling)
            if value < best[0]:
                best = (value, float(xv), float(yv))
    if not np.isfinite(best[0]):
        raise ValueError('no breakdown of the field was located below the search range')
    start = np.array([best[1], best[2]])
    scale = .25 * min(hx, hy)
    simplex = np.array([start, start + [scale, 0.], start + [0., scale]])
    stopping = (dict(xatol=1e-5, fatol=1e-10, maxfev=300) if rough
                else dict(xatol=1e-8, fatol=1e-13, maxfev=3000))
    result = minimize(lambda v: min(_breakdown(phases, bounds, v[0], v[1], ceiling), ceiling),
                      start, method='Nelder-Mead',
                      options=dict(initial_simplex=simplex, **stopping))
    return float(result.fun), float(result.x[0]), float(result.x[1])


def _global_margin(phases, bounds, zmax, nx=24, ny0=48):
    """Smallest y component of the field anywhere in 0 <= z <= zmax."""
    xs = _TWO_PI * np.arange(nx) / nx
    y0s = _TWO_PI * np.arange(ny0) / ny0
    stop, margin = _march(phases, bounds, xs, y0s, zmax / 400., zmax, track_margin=True)
    if np.any(np.isfinite(stop)):
        raise ValueError('the field does not reach the requested height everywhere')
    i, j = np.unravel_index(int(np.argmin(margin)), margin.shape)
    start = np.array([float(xs[i]), float(y0s[j])])
    scale = .25 * min(_TWO_PI / nx, _TWO_PI / ny0)
    simplex = np.array([start, start + [scale, 0.], start + [0., scale]])
    result = minimize(lambda v: _margin_along(phases, bounds, v[0], v[1], zmax), start,
                      method='Nelder-Mead',
                      options=dict(initial_simplex=simplex, xatol=1e-7, fatol=1e-12,
                                   maxfev=600))
    return float(min(float(result.fun), float(margin[i, j])))


def _transported_field(phases, bounds, x, ys, z):
    """The same field at (x, ys, z) from the transported state, for the cross-check."""
    count = 20001
    footpoints = _TWO_PI * np.arange(count) / count
    xpk = (_wave(np.full(count, float(x)), phases[0]), _wave(np.full(count, float(x)), phases[0], 1))
    state = np.stack([footpoints, _azimuth0(phases, bounds, np.full(count, float(x)), footpoints),
                      np.ones(count),
                      _dazimuth0(phases, bounds, np.full(count, float(x)), footpoints)])

    def rhs(zz, flat):
        s = flat.reshape(4, count)
        return np.stack(_transport(phases, bounds, xpk, zz, s[0], s[1], s[2], s[3])).ravel()

    sol = solve_ivp(rhs, (0., float(z)), state.ravel(), method='DOP853', rtol=1e-11, atol=1e-13)
    if not sol.success:
        raise ValueError('the transported family could not be resolved')
    end = sol.y[:, -1].reshape(4, count)
    order = np.argsort(end[0] % _TWO_PI)
    mapped = (end[0] % _TWO_PI)[order]
    azimuth = end[1][order]
    extended_y = np.concatenate([mapped - _TWO_PI, mapped, mapped + _TWO_PI])
    extended_p = np.concatenate([azimuth, azimuth, azimuth])
    queried = np.interp(np.asarray(ys, dtype=float), extended_y, extended_p)
    theta = _theta_pack(phases, bounds,
                        (_wave(np.full(len(ys), float(x)), phases[0]),
                         _wave(np.full(len(ys), float(x)), phases[0], 1)),
                        np.asarray(ys, dtype=float), float(z))[0]
    return np.stack([np.cos(theta), np.sin(theta) * np.sin(queried),
                     np.sin(theta) * np.cos(queried)], axis=-1)


def _oracle_existence_height(phases=None, ceiling=2.0):
    p = _canonical_phases() if phases is None else _check_phases(phases)
    ceiling = float(np.asarray(ceiling, dtype=float))
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('ceiling must be a finite positive scalar')
    bounds = _oracle_seed_bounds(p)
    starts, hx, hy = _global_candidates(p, bounds, ceiling)
    ranked = min(_refine_pair(p, bounds, x0, y0, hx, hy, ceiling, rough=True) for x0, y0 in starts)
    pair = _refine_pair(p, bounds, ranked[1], ranked[2], hx / 8, hy / 8, ceiling)
    height, plane_x = pair[0], pair[1]
    if height >= ceiling * (1 - 1e-12):
        raise ValueError('the field satisfies every stated property throughout the range')

    # The answer is a minimum of the per-plane heights: certify the critical plane and
    # its immediate neighbours, and keep the smallest value any of them reports.
    offsets = (0., -2e-4, 2e-4)
    plane_values = []
    for shift in offsets:
        plane_values.append(_oracle_plane_height(plane_x + shift, p, bounds, ceiling))
    if plane_values[0] > min(plane_values[1], plane_values[2]) + 1e-9:
        raise ValueError('the critical plane is not a local minimum of the plane heights')
    if abs(plane_values[0] - height) > 1e-8:
        raise ValueError('the plane height disagrees with the two-variable minimum')
    height = min([height] + plane_values)

    # The positive branch has to survive strictly below the answer over the WHOLE domain,
    # otherwise the height would be set by loss of that branch instead. The plane the
    # answer comes from need not be the plane where the y component is smallest, so the
    # margin is minimised globally and the critical plane's own margin must not undercut it.
    inner = .999 * height
    plane_x = plane_x % _TWO_PI
    margin = _oracle_branch_margin(plane_x, inner, p, bounds)
    global_margin = min(_global_margin(p, bounds, inner), margin)
    if not np.isfinite(global_margin) or global_margin <= 0:
        raise ValueError('the positive branch does not survive below the returned height')

    # Independent reconstruction of the same field, from the component form rather than
    # the transported state, checked against the prescribed data and the boundary values.
    probe_y = _TWO_PI * np.arange(8) / 8
    inner = .7 * height
    field = _oracle_volume_field(np.array([plane_x]), probe_y, np.array([inner]), p, bounds)
    prescribed = _oracle_prescribed_component(np.array([plane_x])[None, :, None],
                                              probe_y[None, None, :], np.array([inner])[:, None, None],
                                              p, bounds)[..., 0]
    if np.max(np.abs(np.linalg.norm(field, axis=-1) - 1)) > 1e-9:
        raise ValueError('the reconstructed field does not have unit magnitude')
    if np.max(np.abs(field[..., 0] - prescribed)) > 1e-9:
        raise ValueError('the reconstructed field does not carry the prescribed component')
    transported = _transported_field(p, bounds, plane_x, probe_y, inner)
    if np.max(np.abs(field[0, 0] - transported)) > 1e-5:
        raise ValueError('the two reconstructions of the field disagree')
    boundary = _oracle_boundary_field(np.array([plane_x]), probe_y, p, bounds)
    at_zero = _oracle_volume_field(np.array([plane_x]), probe_y, np.array([0.]), p, bounds)[0]
    if not np.allclose(at_zero, boundary, rtol=0, atol=2e-6):
        raise ValueError('the reconstruction does not reproduce the boundary data')
    return float(height)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n\n', 'call': 'existence_height()', 'gold_call': '_oracle_existence_height()'},
     {'setup': 'import numpy as np\np = np.round(np.random.default_rng(909).uniform(0, 2 * np.pi, (3, 3)), 2)\n',
      'call': 'existence_height(p.copy())',
      'gold_call': '_oracle_existence_height(p.copy())'},
     {'setup': 'import numpy as np\np = np.round(np.random.default_rng(101).uniform(0, 2 * np.pi, (3, 4)), 2)\n',
      'call': 'existence_height(p.copy())',
      'gold_call': '_oracle_existence_height(p.copy())'},
     {'setup': 'import numpy as np\n'
               'def case():\n'
               '    try:\n'
               '        existence_height(None, 0.05)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    try:\n'
               '        _oracle_existence_height(None, 0.05)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
