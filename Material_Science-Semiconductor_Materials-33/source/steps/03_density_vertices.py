"""
Construct point-orbital density vertices with symmetric thickness weighting.

In the lattice Bloch gauge, the embedding phases contain the Cartesian

transfer plus each reciprocal vector. Q2D response vertices use the square

root of the one-coordinate thickness averages on each orbital contribution.

Setting those averages to one gives the strict-2D vertices.

Returns
-------
complex weighted density vertices of shape (K, G, 2, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_vertices(
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    orbital_averages: "np.ndarray",
) -> "np.ndarray":
    """Evaluate all band-pair density vertices at a fixed transfer.

    Parameters
    ----------
    left_vectors, right_vectors : "np.ndarray"
        Finite complex arrays (K, 2, 2), K >= 1, indexed by momentum,
        orbital and band, for the states at k and k+transfer, respectively.
        General finite coefficients are supported; physical columns are
        orthonormal Bloch eigenvectors. Inputs are not modified.
    transfer : "np.ndarray"
        Finite real Cartesian vector (2,) in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms; order is kept.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital positions in angstroms.
    orbital_averages : "np.ndarray"
        Finite real array (G, 2), with entries in [0, 1], containing the
        one-coordinate averages from thickness_averages. Use their positive
        square roots as weights in the point-orbital density matrix element
        of exp(-i*(transfer+G).r).

    Returns
    -------
    vertices : "np.ndarray"
        Complex array (K, G, 2, 2), indexed by k, G, left band, right band.
        Band phases are inherited from the supplied coefficients.

    Raises
    ------
    ValueError
        If any documented shape, finiteness, real-input or weight bound is
        violated.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_density_vertices(
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    orbital_averages: "np.ndarray",
) -> "np.ndarray":
    """Point-orbital matrix elements in the stated Fourier convention."""
    left = _finite_array(left_vectors, complex, "left_vectors")
    right = _finite_array(right_vectors, complex, "right_vectors")
    q = _finite_array(transfer, float, "transfer")
    gs = _finite_array(g_vectors, float, "g_vectors")
    tau = _finite_array(centres, float, "centres")
    alpha = _finite_array(orbital_averages, float, "orbital_averages")
    if (
        left.ndim != 3
        or left.shape[1:] != (2, 2)
        or len(left) < 1
        or right.shape != left.shape
    ):
        raise ValueError("coefficient arrays must have equal shape (K, 2, 2)")
    if q.shape != (2,) or gs.ndim != 2 or gs.shape[1] != 2 or len(gs) < 1:
        raise ValueError("invalid transfer or reciprocal-vector shape")
    if tau.shape != (2, 2) or alpha.shape != (len(gs), 2):
        raise ValueError("invalid orbital positions or averages shape")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("orbital averages must lie in [0, 1]")
    phase = np.exp(-1j * ((q + gs) @ tau.T))
    return np.einsum(
        "kan,kam,ga->kgnm", left.conj(), right, phase * np.sqrt(alpha)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Complex phases, the identity vertex, embeddings and invalid weights."""
    return [
        {
            "setup": """import numpy as np

u = np.array(
    [[[1.0, 1.0j], [1.0j, 1.0]], [[1.0, -1.0], [1.0, 1.0]]]
) / np.sqrt(2)
v = np.array([[[1.0, 0.0], [0.0, 1.0j]], [[1.0, 1.0], [-1.0, 1.0]]]) / np.sqrt(
    2
)
q = np.array([0.3, -0.2])
g = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.5]])
t = np.array([[0.0, 0.0], [0.8, 0.35]])
a = np.array([[0.9, 0.8], [0.6, 0.7], [0.4, 0.5]])
""",
            "call": (
                (
                    "density_vertices(u.copy(), v.copy(), q.copy(), "
                    "g.copy(), t.copy(), a.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_density_vertices(u.copy(), v.copy(), "
                    "q.copy(), g.copy(), t.copy(), a.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

u = np.array([[[1.0, 1.0j], [1.0j, 1.0]]]) / np.sqrt(2)
v = u.copy()
q = np.zeros(2)
g = np.zeros((1, 2))
t = np.array([[0.2, -0.4], [0.8, 0.35]])
a = np.ones((1, 2))
""",
            "call": (
                (
                    "density_vertices(u.copy(), v.copy(), q.copy(), "
                    "g.copy(), t.copy(), a.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_density_vertices(u.copy(), v.copy(), "
                    "q.copy(), g.copy(), t.copy(), a.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

u = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)
v = np.array([[[0.6, -0.8j], [-0.8j, 0.6]]])
q = np.array([1.1, -0.8])
g = np.array([[0.5, 0.7], [-0.5, -0.7]])
t = np.array([[0.3, -0.6], [1.2, 0.9]])
a = np.array([[0.0, 0.4], [0.7, 1.0]])
""",
            "call": (
                (
                    "density_vertices(u.copy(), v.copy(), q.copy(), "
                    "g.copy(), t.copy(), a.copy())"
                )
            ),
            "gold_call": (
                (
                    "_oracle_density_vertices(u.copy(), v.copy(), "
                    "q.copy(), g.copy(), t.copy(), a.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(
            np.eye(2)[None],
            np.eye(2)[None],
            np.zeros(2),
            np.zeros((1, 2)),
            np.zeros((2, 2)),
            np.array([[1.0, -1.0]]),
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(density_vertices)",
            "gold_call": "rejected(_oracle_density_vertices)",
        },
    ]
