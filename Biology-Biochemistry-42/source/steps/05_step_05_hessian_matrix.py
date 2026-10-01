"""
Assemble the 3M x 3M Hessian of the elastic network from the bead coordinates and the spring constant matrix, following equation (5) of the source for the off-diagonal three-by-three sub-blocks and completing each diagonal sub-block so that a rigid translation of the whole network costs no energy. Pairs with no spring contribute nothing.

Normal mode analysis of an elastic network needs the second derivatives of the harmonic potential. For a spring between two beads the resulting three-by-three block depends only on the spring constant, the inter-bead distance and the direction cosines of the connecting vector, which is why the source can write it in closed form. No energy minimisation is required, because the reference conformation is assumed to sit at the potential minimum.

Returns
-------
np.ndarray of shape (3M, 3M), float64: the symmetric Hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hessian_matrix(coords: "np.ndarray", springs: "np.ndarray") -> "np.ndarray":
    """Assemble the 3M x 3M Hessian of the elastic network from the bead coordinates and
    the spring constant matrix, following equation (5) of the source for the off-
    diagonal three-by-three sub-blocks and completing each diagonal sub-block so that a
    rigid translation of the whole network costs no energy. Pairs with no spring
    contribute nothing.

    Parameters
    ----------
    coords : np.ndarray of shape (M, 3)
        bead coordinates in angstrom of the network under analysis.
    springs : np.ndarray of shape (M, M)
        symmetric spring constants in kcal/(mol angstrom^2); a zero entry means
        the two beads are not connected.

    Returns
    -------
    np.ndarray of shape (3M, 3M), float64: the symmetric Hessian

    Raises
    ------
    ValueError
        if coords does not have shape (M, 3), if springs is not square, does not
        match coords or is not symmetric, or if two beads joined by a spring sit
        at the same position.
    """
    return hessian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hessian_matrix(coords: "np.ndarray", springs: "np.ndarray") -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    springs = np.asarray(springs, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (M, 3)")
    if springs.shape != (len(coords), len(coords)):
        raise ValueError("springs must be square and match coords")
    if not np.allclose(springs, springs.T, rtol=0.0, atol=1e-12):
        raise ValueError("springs must be symmetric")
    n = len(coords)
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or springs[i, j] == 0.0:
                continue
            dr = coords[j] - coords[i]
            d2 = float(dr @ dr)
            if d2 <= 0.0:
                raise ValueError("two connected beads share a position")
            blk = -(springs[i, j] / d2) * np.outer(dr, dr)
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blk
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
k_na = _oracle_nucleic_spring_matrix(seq, coords, classes)
""",
            "call": "hessian_matrix(coords, k_na)",
            "gold_call": "_oracle_hessian_matrix(coords, k_na)",
        },
        {
            "setup": """import numpy as np
coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
springs = np.array([[0.0, 2.5], [2.5, 0.0]])
""",
            "call": "hessian_matrix(coords, springs)",
            "gold_call": "_oracle_hessian_matrix(coords, springs)",
        },
        {
            "setup": """import numpy as np
coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, -1.0, 0.5],
                   [-1.5, 0.25, 2.0]])
springs = np.zeros((4, 4))
springs[0, 1] = springs[1, 0] = 1e-6
springs[2, 3] = springs[3, 2] = 1e6
""",
            "call": "hessian_matrix(coords, springs)",
            "gold_call": "_oracle_hessian_matrix(coords, springs)",
        },
        {
            "setup": """import numpy as np
coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
springs = np.array([[0.0, 2.5], [1.0, 0.0]])
def run_model():
    try:
        hessian_matrix(coords, springs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hessian_matrix(coords, springs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
