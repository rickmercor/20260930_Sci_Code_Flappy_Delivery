"""
Height to which the admissible field survives above a single periodic plane.

Height to which the admissible field survives above a single periodic plane.

The admissible field of the preceding steps need not exist for every height: above a
given plane x=x0 the prescribed data can be carried upward only so far before the field
with all of the stated properties ceases to exist there. This step reports that height
for one plane, as a supremum over the periodic y line.

Returns
-------
The supremum of heights reached by the admissible field above the plane x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plane_height(x, phases, bounds, ceiling=2.0):
    """Return the supremum of heights reached by the admissible field above the plane x.

    x is a finite real scalar; phases and bounds follow the seed_bounds conventions;
    ceiling is a finite positive search limit. The returned float is the largest h such
    that the field with all stated properties exists above this plane for every height
    below h.

    Raises
    ------
    ValueError
        If any input is malformed or nonfinite, if ceiling is not positive, or if the
        field above this plane still satisfies every stated property at ceiling.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar

_TWO_PI = 2 * np.pi


def _plane_candidates(phases, bounds, x, ceiling, family):
    """Footpoints whose transported state stops earliest on a coarse march."""
    y0s = _TWO_PI * np.arange(family) / family
    stop, _ = _march(phases, bounds, [x], y0s, min(ceiling, 1.0) / 250., ceiling)
    row = stop[0]
    finite = np.isfinite(row)
    if not finite.any():
        return y0s, None
    return y0s, np.argsort(np.where(finite, row, np.inf))[:3]


def _oracle_plane_height(x, phases, bounds, ceiling=2.0):
    phases = _check_phases(phases)
    bounds = _check_bounds(bounds)
    x, ceiling = _two_scalars(x, ceiling)
    if ceiling <= 0:
        raise ValueError('ceiling must be positive')
    family = 192
    y0s, order = _plane_candidates(phases, bounds, x, ceiling, family)
    if order is None:
        raise ValueError('the field above this plane survives the whole search range')
    spacing = _TWO_PI / family
    best = np.inf
    for j in order:
        fun = lambda t: min(_breakdown(phases, bounds, x, t, ceiling), ceiling)
        node = float(y0s[j])
        res = minimize_scalar(fun, bounds=(node - spacing, node + spacing), method='bounded',
                              options={'xatol': 1e-11})
        best = min(best, float(res.fun), fun(node))
    if not np.isfinite(best) or best >= ceiling * (1 - 1e-12):
        raise ValueError('the field above this plane survives the whole search range')
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n',
      'call': 'plane_height(2.2143518802, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_plane_height(2.2143518802, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n',
      'call': 'plane_height(1.0, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_plane_height(1.0, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\np = np.round(np.random.default_rng(101).uniform(0, 2 * np.pi, (3, 4)), 2)\n',
      'call': 'plane_height(2.5, p.copy(), seed_bounds(p.copy()))',
      'gold_call': '_oracle_plane_height(2.5, p.copy(), _oracle_seed_bounds(p.copy()))'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n',
      'call': 'plane_height(2.2143518802, p.copy(), seed_bounds(p.copy()), 0.5)',
      'gold_call': '_oracle_plane_height(2.2143518802, p.copy(), _oracle_seed_bounds(p.copy()), 0.5)'},
     {'setup': 'import numpy as np\n'
               'p = np.array([[5.93, 2.26, 4.93, 3.72, 1.85], [5.8, 5.46, 2.29, 6.11, 1.41], [5.06, 4.28, 2.96, '
               '0.19, 5.62]])\n'
               '\n'
               'def case():\n'
               '    _case_bounds = seed_bounds(p.copy())\n'
               '    try:\n'
               '        plane_height(2.2143518802, p.copy(), _case_bounds, 0.05)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    _case_bounds = _oracle_seed_bounds(p.copy())\n'
               '    try:\n'
               '        _oracle_plane_height(2.2143518802, p.copy(), _case_bounds, 0.05)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
