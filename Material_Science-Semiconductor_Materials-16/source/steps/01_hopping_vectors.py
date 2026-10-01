"""
Return the fifteen bond vectors of a 1T-MX2 monolayer, stacked in one array in this order: the six metal-chalcogen bonds, then three in-plane nearest-neighbour vectors, then three in-plane next-nearest-neighbour vectors, then three vectors reaching from one chalcogen plane to the other. Within each of those four families, list the vectors in order of increasing azimuthal angle about the metal site, measured from the positive x axis. Only half of each in-plane family is listed, since a later step supplies the opposite members. The source fixes a convention here that the natural reading does not; follow the source.

The 1T structure stacks a triangular metal plane between two chalcogen planes. Its geometry is fixed by the in-plane lattice constant together with the angle the metal-chalcogen bond makes with the plane, and every hopping amplitude follows from those two numbers. Both the length of the metal-chalcogen bond and the reach of the two chalcogen planes are set by that angle, and more than one plausible relation exists for each.

Returns
-------
ndarray of shape (15, 3), the bond vectors in Angstrom, in the order described.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hopping_vectors(a: float, theta: float) -> "np.ndarray":
    """Return the fifteen bond vectors of a 1T-MX2 monolayer, stacked in one array in this order: the six metal-chalcogen bonds, then three in-plane nearest-neighbour vectors, then three in-plane next-nearest-neighbour vectors, then three vectors reaching from one chalcogen plane to the other. Within each of those four families, list the vectors in order of increasing azimuthal angle about the metal site, measured from the positive x axis. Only half of each in-plane family is listed, since a later step supplies the opposite members. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (15, 3), the bond vectors in Angstrom, in the order described.

    Raises
    ------
    ValueError: if the lattice constant is not positive, or if the bond angle does not lie strictly between 0 and pi/2.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_hopping_vectors(a: float, theta: float) -> "np.ndarray":
    if a <= 0:
        raise ValueError("lattice constant must be positive")
    if not (0.0 < theta < 0.5 * np.pi):
        raise ValueError("bond angle must lie strictly between 0 and pi/2")
    c, s = np.cos(theta), np.sin(theta)
    r3 = np.sqrt(3.0)
    # CONVENTION (source, Table I): bond length a/(sqrt(3) cos t), not a/cos t.
    b = a / (r3 * c)
    # LISTING ORDER, fixed by the problem statement rather than by the physics: every family
    # runs by increasing azimuthal angle about +x. Without it the six M-X bonds admit several
    # equally defensible listings and the step is not well posed.
    A = np.array([[r3 / 2 * c, 0.5 * c, s], [0.0, c, -s], [-r3 / 2 * c, 0.5 * c, s],
                  [-r3 / 2 * c, -0.5 * c, -s], [0.0, -c, s], [r3 / 2 * c, -0.5 * c, -s]]) * b
    # the three independent in-plane NN directions; their opposites are supplied by the
    # cosine sum in step 6 rather than being listed here.
    D = np.array([[1.0, 0.0, 0.0], [0.5, r3 / 2, 0.0], [-0.5, r3 / 2, 0.0]]) * a
    # CONVENTION (source, Fig. 2): next-nearest in-plane neighbour at a*sqrt(3), not a.
    C = np.array([[r3 / 2, 0.5, 0.0], [0.0, 1.0, 0.0], [-r3 / 2, 0.5, 0.0]]) * (a * r3)
    # CONVENTION (source, App. B): inter-plane vector carries 2*sin(theta); the printed
    # factor d = cos^2 + 4 sin^2 is the signature of that two.
    L = np.array([[r3 / 2 * c, 0.5 * c, -2 * s], [-r3 / 2 * c, 0.5 * c, -2 * s],
                  [0.0, -c, -2 * s]]) * b
    return np.vstack([A, D, C, L])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "hopping_vectors(3.69, 0.5965)",
         "gold_call": "_oracle_hopping_vectors(3.69, 0.5965)"},   # normal
        {"setup": "import numpy as np",
         "call": "hopping_vectors(1.0, 0.7853981633974483)",
         "gold_call": "_oracle_hopping_vectors(1.0, 0.7853981633974483)"},   # boundary
        {"setup": "import numpy as np",
         "call": "hopping_vectors(3.652, 0.5993)",
         "gold_call": "_oracle_hopping_vectors(3.652, 0.5993)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        hopping_vectors(3.69, 1.5707963267948966)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_hopping_vectors(3.69, 1.5707963267948966)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
