"""
Return the squared magnitude of the normalized overlap between a complex field and a target profile sampled on the same uniform grid of spacing dx: the squared magnitude of the integral of the conjugated target times the field, divided by the product of the two fields' powers. Raise ValueError if the two arrays have different shapes, if dx is not strictly positive, or if either field carries no power.

Two beams can share a width and still differ in shape, and a measure built from the width alone cannot see that difference. Projecting one field onto the other and normalizing by both powers gives a number that is one only when the two agree up to an overall constant factor, and that falls as soon as their shapes part company. It is the standard way to ask how close an engineered final state came to the state it was aiming at, and it is sensitive to structure a single collective coordinate cannot carry.

Returns
-------
float: the normalized squared overlap, between zero and one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overlap_fidelity(u: 'np.ndarray', target: 'np.ndarray', dx: float) -> float:
    """Return the squared magnitude of the normalized overlap between a complex field and a target profile sampled on the same uniform grid of spacing dx: the squared magnitude of the integral of the conjugated target times the field, divided by the product of the two fields' powers. Raise ValueError if the two arrays have different shapes, if dx is not strictly positive, or if either field carries no power.

    Returns
    -------
    float: the normalized squared overlap, between zero and one.

    Raises
    ------
    ValueError
        If the shapes differ, dx is not strictly positive, or either field carries no power.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_overlap_fidelity(u: "np.ndarray", target: "np.ndarray", dx: float) -> float:
    u = np.asarray(u, dtype=complex); target = np.asarray(target, dtype=complex)
    dx = float(dx)
    if u.shape != target.shape:
        raise ValueError("u and target must have the same shape")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    nu = np.sum(np.abs(u) ** 2) * dx
    nt = np.sum(np.abs(target) ** 2) * dx
    if nu <= 0.0 or nt <= 0.0:
        raise ValueError("both fields must carry positive power")
    ov = np.sum(np.conj(target) * u) * dx
    return float(np.abs(ov) ** 2 / (nu * nt))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step test specifications."""
    return [{'setup': 'import numpy as np\n'
               'g = _grid(256, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = np.exp(-x**2/2).astype(complex)\n'
               't = np.exp(-x**2/2)',
      'call': 'overlap_fidelity(u.copy(), t.copy(), dx)',
      'gold_call': '_oracle_overlap_fidelity(u, t, dx)'},
     {'setup': 'import numpy as np\n'
               'g = _grid(256, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = (2.5*np.exp(-x**2/(2*0.9**2))).astype(complex)\n'
               't = np.exp(-x**2/(2*1.4**2))',
      'call': 'overlap_fidelity(u.copy(), t.copy(), dx)',
      'gold_call': '_oracle_overlap_fidelity(u, t, dx)'},
     {'setup': 'import numpy as np\n'
               'g = _grid(256, 20.0)\n'
               'x = g[0]; dx = x[1]-x[0]\n'
               'u = (np.exp(-x**2/2)*np.exp(1j*0.7*x)).astype(complex)\n'
               't = np.exp(-x**2/2)',
      'call': 'overlap_fidelity(u.copy(), t.copy(), dx)',
      'gold_call': '_oracle_overlap_fidelity(u, t, dx)'},
     {'setup': 'import numpy as np\n'
               'x = _grid(256, 20.0)[0]\n'
               'dx = x[1] - x[0]\n'
               'u = (3.7 * np.exp(-x**2/2)).astype(complex)\n'
               't = np.exp(-x**2/(2*1.3**2))',
      'call': 'overlap_fidelity(u.copy(), t.copy(), dx)',
      'gold_call': '_oracle_overlap_fidelity(u.copy(), t.copy(), dx)'},
     {'setup': 'import numpy as np\n'
               'def probe_public():\n'
               '    try:\n'
               '        overlap_fidelity(np.zeros(8, dtype=complex), np.ones(8), 0.25)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def probe_gold():\n'
               '    try:\n'
               '        _oracle_overlap_fidelity(np.zeros(8, dtype=complex), np.ones(8), 0.25)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'probe_public()',
      'gold_call': 'probe_gold()'}]
