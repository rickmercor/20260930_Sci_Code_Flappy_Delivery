"""
Build the two eigenstrain directions that the source attaches to the non-smooth strength criterion, Eq. (17b), from the current total strain. The first direction carries the volumetric response and the second the shape-changing response. Both are written in the source in a specific scaled form, stated in the sentence that follows Eq. (17); reproduce that form exactly as given, because it is what fixes the meaning of each multiplier.

Eq. (14) and Eq. (15) tie the fracture eigenstrain to a small set of fixed directions multiplied by scalar magnitudes, so that only those scalars have to be solved for at a point. For the non-smooth criterion the source splits that set into two independent facets, checked and activated separately. The directions are taken from the current total strain so the eigenstrain always acts to relieve it, which fixes their signs.

Returns
-------
A (2, 6) float64 array whose rows are the volumetric and the shape-changing direction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def faceted_directions(eps: "np.ndarray") -> "np.ndarray":
    """Build the two eigenstrain directions that the source attaches to the non-smooth strength
    criterion, Eq. (17b), from the current total strain, in the scaled form the source states
    in the sentence that follows Eq. (17).

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.

    Returns:
        A (2, 6) float64 array whose rows are the volumetric and the shape-changing direction.
        When the trace of eps is exactly zero the volumetric direction is the zero vector, and
        when the deviatoric part of eps vanishes the shape-changing direction is the zero vector.

    Raises:
        ValueError: If eps is not a finite (6,) array.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def _unit(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0.0 else np.zeros(6)


def _oracle_faceted_directions(eps: "np.ndarray") -> "np.ndarray":
    eps = np.asarray(eps, dtype=float)
    if eps.shape != (6,) or not np.all(np.isfinite(eps)):
        raise ValueError("eps must be a finite six-component array")
    t = _tr(eps)
    s = (t / abs(t)) if t != 0.0 else 0.0
    G1 = (1.0 / 3.0) * s * _im()
    G2 = _unit(_dev(eps))
    return np.vstack([G1, G2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\n',
         'call': 'faceted_directions(eps)',
         'gold_call': '_oracle_faceted_directions(eps)'},
        {'setup': 'import numpy as np\neps = np.array([-1.2e-3, -1.3e-3, 0.0, 4.0e-4, 0.0, 2.0e-4])\n',
         'call': 'faceted_directions(eps)',
         'gold_call': '_oracle_faceted_directions(eps)'},
        {'setup': 'import numpy as np\neps = np.array([8.0e-4, -8.0e-4, 0.0, 0.0, 0.0, 1.5e-3])\n',
         'call': 'faceted_directions(eps)',
         'gold_call': '_oracle_faceted_directions(eps)'},
        {'setup': 'import numpy as np\n# invalid input: a five-component strain is not in the six-component layout and must raise ValueError\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 9.0e-4])\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: faceted_directions(eps))',
         'gold_call': '_catches_value_error(lambda: _oracle_faceted_directions(eps))'},
    ]
