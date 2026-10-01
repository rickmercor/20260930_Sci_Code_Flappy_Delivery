"""
Return the two primitive reciprocal lattice vectors of the triangular lattice followed by the zone centre, the edge midpoint and the zone corner, stacked in one array in that order, for the real-space primitive vectors given in the problem statement. The source fixes a convention here that the natural reading does not; follow the source.

Every momentum-space quantity in this task is either sampled on a uniform grid spanned by the reciprocal lattice vectors or evaluated at one of the three high-symmetry points, so both must come from the same convention for the real-space cell.

Returns
-------
ndarray of shape (5, 3), in inverse Angstrom: the two reciprocal lattice vectors, then the three high-symmetry points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def brillouin_zone(a: float) -> "np.ndarray":
    """Return the two primitive reciprocal lattice vectors of the triangular lattice followed by the zone centre, the edge midpoint and the zone corner, stacked in one array in that order, for the real-space primitive vectors given in the problem statement. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (5, 3), in inverse Angstrom: the two reciprocal lattice vectors, then the three high-symmetry points.

    Raises
    ------
    ValueError: if the lattice constant is not positive.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_brillouin_zone(a: float) -> "np.ndarray":
    if a <= 0:
        raise ValueError("lattice constant must be positive")
    r3 = np.sqrt(3.0)
    b1 = 2 * np.pi / a * np.array([1.0, -1.0 / r3, 0.0])
    b2 = 2 * np.pi / a * np.array([0.0, 2.0 / r3, 0.0])
    # CONVENTION: |M| = 2 pi / (sqrt(3) a) and |K| = 4 pi / (3 a).
    return np.vstack([b1, b2, np.zeros(3), 0.5 * b1, (2.0 * b1 + b2) / 3.0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "brillouin_zone(3.69)",
         "gold_call": "_oracle_brillouin_zone(3.69)"},   # normal
        {"setup": "import numpy as np",
         "call": "brillouin_zone(1.0)",
         "gold_call": "_oracle_brillouin_zone(1.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "brillouin_zone(3.652)",
         "gold_call": "_oracle_brillouin_zone(3.652)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        brillouin_zone(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_brillouin_zone(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
