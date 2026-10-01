"""
Return the prescribed beam width and its second derivative with respect to propagation distance at the given distance, for the smooth interpolation the source adopts between the initial and final widths over a total distance zf. The interpolation is the quintic in the normalized distance whose value matches the endpoints and whose first and second derivatives vanish at both of them. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive.

Prescribing the width as a function of distance and then asking what medium would produce it is the inverse of the usual problem. The interpolation must join the two equilibrium widths with vanishing slope and vanishing curvature at both ends, so that the protocol starts and finishes at a genuine stationary state of the reduced description rather than leaving the beam oscillating.

Returns
-------
ndarray of shape (2,), float64: the prescribed width and its second derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimum_jerk_width(z: float, zf: float, a_i: float, a_f: float) -> 'np.ndarray':
    """Return the prescribed beam width and its second derivative with respect to propagation distance at the given distance, for the smooth interpolation the source adopts between the initial and final widths over a total distance zf. The interpolation is the quintic in the normalized distance whose value matches the endpoints and whose first and second derivatives vanish at both of them. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive.

    Returns
    -------
    ndarray of shape (2,), float64: the prescribed width and its second derivative.

    Raises
    ------
    ValueError
        If zf is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_minimum_jerk_width(z: float, zf: float, a_i: float, a_f: float) -> "np.ndarray":
    z = float(z); zf = float(zf)
    if zf <= 0.0:
        raise ValueError("zf must be positive")
    s = min(max(z / zf, 0.0), 1.0)
    a = a_i + (a_f - a_i) * (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)
    add = (a_f - a_i) * (60.0 * s - 180.0 * s ** 2 + 120.0 * s ** 3) / zf ** 2
    return np.array([a, add], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step test specifications."""
    return [{'setup': 'import numpy as np',
      'call': 'minimum_jerk_width(1.1, 2.2, 0.8928089, 0.4610400)',
      'gold_call': '_oracle_minimum_jerk_width(1.1, 2.2, 0.8928089, 0.4610400)'},
     {'setup': 'import numpy as np',
      'call': 'minimum_jerk_width(0.0, 2.2, 0.8928089, 0.4610400)',
      'gold_call': '_oracle_minimum_jerk_width(0.0, 2.2, 0.8928089, 0.4610400)'},
     {'setup': 'import numpy as np',
      'call': 'minimum_jerk_width(5.0, 2.2, 0.8928089, 0.4610400)',
      'gold_call': '_oracle_minimum_jerk_width(5.0, 2.2, 0.8928089, 0.4610400)'},
     {'setup': 'import numpy as np',
      'call': 'minimum_jerk_width(0.55, 2.2, 0.8928089, 0.4610400)',
      'gold_call': '_oracle_minimum_jerk_width(0.55, 2.2, 0.8928089, 0.4610400)'},
     {'setup': 'import numpy as np',
      'call': 'minimum_jerk_width(0.4, 2.0, 1.0, 1.0)',
      'gold_call': '_oracle_minimum_jerk_width(0.4, 2.0, 1.0, 1.0)'},
     {'setup': 'import numpy as np\n'
               'def probe_public():\n'
               '    try:\n'
               '        minimum_jerk_width(1.0, 0.0, 1.0, 0.4)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def probe_gold():\n'
               '    try:\n'
               '        _oracle_minimum_jerk_width(1.0, 0.0, 1.0, 0.4)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'probe_public()',
      'gold_call': 'probe_gold()'}]
