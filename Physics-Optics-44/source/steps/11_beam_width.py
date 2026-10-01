"""
Return the beam width of the given field on the given coordinate grid, defined so that a Gaussian intensity profile of the form used by the reduced description returns exactly its own width parameter. Measure the spread about the field's own intensity centroid, not about the origin. Raise ValueError if the field and the grid have different shapes, if the spacing is not strictly positive, or if the field carries no power.

Comparing a simulated field with the reduced description needs a width that means the same thing for both. Referring the spread to the intensity centroid removes the drift that third-order effects and any asymmetry of the control profile introduce, so that what is compared is the breathing the protocol is meant to control.

Returns
-------
float: the beam width of the field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def beam_width(u: 'np.ndarray', x: 'np.ndarray', dx: float) -> float:
    """Return the beam width of the given field on the given coordinate grid, defined so that a Gaussian intensity profile of the form used by the reduced description returns exactly its own width parameter. Measure the spread about the field's own intensity centroid, not about the origin. Raise ValueError if the field and the grid have different shapes, if the spacing is not strictly positive, or if the field carries no power.

    Returns
    -------
    float: the beam width of the field.

    Raises
    ------
    ValueError
        If u and x have different shapes, dx is not strictly positive, or the field carries no power.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_beam_width(u: "np.ndarray", x: "np.ndarray", dx: float) -> float:
    u = np.asarray(u, dtype=complex); x = np.asarray(x, dtype=float)
    dx = float(dx)
    if u.shape != x.shape:
        raise ValueError("u and x must have the same shape")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    a2 = np.abs(u) ** 2
    p = np.sum(a2) * dx
    if p <= 0.0:
        raise ValueError("the field must carry positive power")
    xc = np.sum(x * a2) * dx / p
    return float(np.sqrt(2.0 * np.sum((x - xc) ** 2 * a2) * dx / p))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step test specifications."""
    return [{'setup': 'import numpy as np\n'
               'g = _grid(512, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = np.exp(-x**2/(2*0.7**2)).astype(complex)',
      'call': 'beam_width(u.copy(), x.copy(), dx)',
      'gold_call': '_oracle_beam_width(u, x, dx)'},
     {'setup': 'import numpy as np\n'
               'g = _grid(512, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = (2.5*np.exp(-(x-3.0)**2/(2*1.3**2))).astype(complex)',
      'call': 'beam_width(u.copy(), x.copy(), dx)',
      'gold_call': '_oracle_beam_width(u, x, dx)'},
     {'setup': 'import numpy as np\n'
               'g = _grid(512, 60.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = np.exp(-x**2/(2*8.0**2)).astype(complex)',
      'call': 'beam_width(u.copy(), x.copy(), dx)',
      'gold_call': '_oracle_beam_width(u, x, dx)'},
     {'setup': 'import numpy as np\nx = np.arange(8, dtype=float)\nu = np.zeros(8, dtype=complex)\nu[3] = 1.0',
      'call': 'beam_width(u.copy(), x.copy(), 1.0)',
      'gold_call': '_oracle_beam_width(u.copy(), x.copy(), 1.0)'},
     {'setup': 'import numpy as np\n'
               'g = _grid(512, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = 1.7*np.exp(-(x-1.25)**2/(2*1.1**2))*np.exp(1j*0.8*x**2)',
      'call': 'beam_width(u.copy(), x.copy(), dx)',
      'gold_call': '_oracle_beam_width(u, x, dx)'},
     {'setup': 'import numpy as np\n'
               'def probe_public():\n'
               '    try:\n'
               '        beam_width(np.zeros(8, dtype=complex), _grid(8, 1.0)[0], 0.25)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def probe_gold():\n'
               '    try:\n'
               '        _oracle_beam_width(np.zeros(8, dtype=complex), _grid(8, 1.0)[0], 0.25)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'probe_public()',
      'gold_call': 'probe_gold()'}]
