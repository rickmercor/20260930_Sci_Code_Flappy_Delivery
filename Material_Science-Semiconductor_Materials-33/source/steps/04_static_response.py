"""
Build the static insulating response from both interband channels.

All valence states are occupied and all conduction states are empty. No

time-reversal reduction is assumed. Weighted vertices give the auxiliary

response used by the symmetric Q2D dielectric construction; unweighted ones

give the strict-2D irreducible response. The k mesh has equal weights.

Returns
-------
complex static response matrix of shape (G, G).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_response(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    vertices: "np.ndarray",
    spin_degeneracy: int = 1,
) -> "np.ndarray":
    """Evaluate the zero-temperature static interband spectral sum.

    Parameters
    ----------
    left_energies, right_energies : "np.ndarray"
        Finite real arrays (K, 2), K >= 1, in eV, for k and k+q.
        Column 0 is occupied valence and column 1 empty conduction.
        Each row is strictly increasing, and both cross-momentum
        conduction-minus-valence gaps must be strictly positive.
    vertices : "np.ndarray"
        Finite complex array (K, G, 2, 2), G >= 1, from density_vertices.
    spin_degeneracy : int
        Positive integer multiplicity. One describes the spinless benchmark.
        It is independent of the two occupied/empty transition directions.

    Returns
    -------
    response : "np.ndarray"
        Complex Hermitian array (G, G) in inverse eV. Use the occupation
        difference divided by the left-minus-right energy difference for
        each of the two interband directions, an equal 1/K mesh weight,
        and the specified spin multiplicity. Intraband terms vanish.

    Raises
    ------
    ValueError
        If shapes, finite/real energy inputs, energy ordering, cross gaps,
        or the positive integer multiplicity violate this contract.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_static_response(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    vertices: "np.ndarray",
    spin_degeneracy: int = 1,
) -> "np.ndarray":
    """Full two-channel response; no assumed equality of the two channels."""
    left = _finite_array(left_energies, float, "left_energies")
    right = _finite_array(right_energies, float, "right_energies")
    vertex = _finite_array(vertices, complex, "vertices")
    spin = _integer_scalar(spin_degeneracy, "spin_degeneracy", 1)
    if (
        left.ndim != 2
        or left.shape[1] != 2
        or len(left) < 1
        or right.shape != left.shape
    ):
        raise ValueError("energies must have equal shape (K, 2)")
    if (
        vertex.ndim != 4
        or vertex.shape[0] != len(left)
        or vertex.shape[1] < 1
        or vertex.shape[2:] != (2, 2)
    ):
        raise ValueError("vertices must have shape (K, G, 2, 2)")
    gap01 = right[:, 1] - left[:, 0]
    gap10 = left[:, 1] - right[:, 0]
    if (
        np.any(np.diff(left, axis=1) <= 0)
        or np.any(np.diff(right, axis=1) <= 0)
        or np.any(gap01 <= 0)
        or np.any(gap10 <= 0)
    ):
        raise ValueError(
            "all within- and cross-momentum gaps must be positive"
        )
    v01, v10 = vertex[:, :, 0, 1], vertex[:, :, 1, 0]
    response = -np.einsum("kg,kh,k->gh", v01, v01.conj(), 1 / gap01)
    response -= np.einsum("kg,kh,k->gh", v10, v10.conj(), 1 / gap10)
    return spin * response / len(left)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """
    Unequal channels, complex local fields, a null vertex and spin scaling.
    """
    return [
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 2.0]])
f = np.array([[-2.0, 3.0]])
v = np.array([[[[0.0, 1.0 + 0.3j], [0.4 - 0.2j, 0.0]]]])
s = 1
""",
            "call": ("static_response(e.copy(), f.copy(), v.copy(), " "s)"),
            "gold_call": (
                "_oracle_static_response(e.copy(), f.copy(), " "v.copy(), s)"
            ),
        },
        {
            "setup": """import numpy as np

e = np.array([[-1.2, 1.8], [-0.8, 2.3]])
f = np.array([[-1.1, 2.2], [-1.4, 1.9]])
rng = np.random.default_rng(4302)
v = rng.normal(size=(2, 3, 2, 2)) + 1j * rng.normal(size=(2, 3, 2, 2))
s = 2
""",
            "call": ("static_response(e.copy(), f.copy(), v.copy(), " "s)"),
            "gold_call": (
                "_oracle_static_response(e.copy(), f.copy(), " "v.copy(), s)"
            ),
        },
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 1.0], [-2.0, 2.0]])
f = e.copy()
v = np.broadcast_to(np.eye(2), (2, 1, 2, 2)).copy()
s = 1
""",
            "call": ("static_response(e.copy(), f.copy(), v.copy(), " "s)"),
            "gold_call": (
                "_oracle_static_response(e.copy(), f.copy(), " "v.copy(), s)"
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            np.ones((1, 1, 2, 2)),
            1,
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(static_response)",
            "gold_call": "rejected(_oracle_static_response)",
        },
    ]
