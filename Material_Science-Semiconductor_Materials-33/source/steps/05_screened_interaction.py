"""
Invert the symmetric dielectric matrix and form the screened interaction.

The supplied response already contains the orbital-resolved square-root

weights for Q2D. The dielectric Coulomb factors use the strict-2D potential;

the external factors of W use the separate double-coordinate average.

At an exactly zero momentum magnitude, the benchmark removes the head and

wings of W while retaining the screened body. No elementwise inversion is used.

Returns
-------
a complex array (3, G, G) holding epsilon, inverse, and W.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_interaction(
    response: "np.ndarray",
    magnitudes: "np.ndarray",
    area: float,
    coupling: float,
    pair_averages: "np.ndarray",
) -> "np.ndarray":
    """Return the dielectric matrix, its inverse and screened interaction.

    Parameters
    ----------
    response : "np.ndarray"
        Finite complex array (G, G), G >= 1, Hermitian to absolute entrywise
        tolerance 1e-10 and negative semidefinite to eigenvalue tolerance
        1e-10, in inverse eV. Roundoff within that tolerance is symmetrized.
        This is the weighted response from static_response.
    magnitudes : "np.ndarray"
        Finite nonnegative real vector (G,) of |q+G| in inverse angstroms.
        A zero entry invokes the exact zero-mode convention below.
    area : float
        Positive finite unit-cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms; the strict-2D
        Fourier interaction at positive p is 2*pi*C/(area*p).
    pair_averages : "np.ndarray"
        Finite real vector (G,) in [0, 1], the double-coordinate averages
        from thickness_averages. Use ones for strict 2D.

    Returns
    -------
    matrices : "np.ndarray"
        Complex array (3, G, G), ordered as the symmetric dielectric matrix,
        its full matrix inverse, and W in eV. Q2D uses the source's symmetric
        construction with the supplied weighted response and pair averages.
        Each p=0 row/column of the dielectric matrix and its inverse equals
        the corresponding identity row/column; that row/column of W is zero.
        The body is computed with all remaining local-field couplings.

    Raises
    ------
    ValueError
        If shapes, finiteness, real inputs, signs, Hermiticity, response
        eigenvalue bounds, average bounds, or positive area are violated.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_screened_interaction(
    response: "np.ndarray",
    magnitudes: "np.ndarray",
    area: float,
    coupling: float,
    pair_averages: "np.ndarray",
) -> "np.ndarray":
    """Symmetric Q2D dielectric inversion with separate external factors."""
    chi = _finite_array(response, complex, "response")
    p = _finite_array(magnitudes, float, "magnitudes")
    beta = _finite_array(pair_averages, float, "pair_averages")
    cell_area = _finite_scalar(area, "area")
    constant = _finite_scalar(coupling, "coupling")
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0):
        raise ValueError("magnitudes must be a nonempty nonnegative vector")
    if chi.shape != (len(p), len(p)) or beta.shape != p.shape:
        raise ValueError("response or pair-average shape mismatch")
    if cell_area <= 0 or np.any(beta < 0) or np.any(beta > 1):
        raise ValueError("area must be positive and averages in [0, 1]")
    if not np.allclose(chi, chi.T.conj(), rtol=0, atol=1e-10):
        raise ValueError("response must be Hermitian")
    chi = (chi + chi.T.conj()) / 2
    if np.linalg.eigvalsh(chi)[-1] > 1e-10:
        raise ValueError("response must be negative semidefinite")
    v = np.zeros(len(p))
    positive = p > 0
    v[positive] = 2 * np.pi * constant / (cell_area * p[positive])
    root = np.sqrt(v)
    epsilon = np.eye(len(p)) - root[:, None] * chi * root[None, :]
    inverse = np.linalg.solve(epsilon, np.eye(len(p)))
    outer_root = np.sqrt(v * beta)
    screened = outer_root[:, None] * inverse * outer_root[None, :]
    return np.stack((epsilon, inverse, screened))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """
    Local-field mixing, vacuum, retained zero-mode body and zero coupling.
    """
    return [
        {
            "setup": """import numpy as np

b = np.array([[0.3, 0.1j], [-0.2j, 0.4], [0.2 + 0.1j, -0.3]])
chi = -b @ b.conj().T
p = np.array([0.13, 1.2, 2.1])
beta = np.array([0.85, 0.4, 0.22])
area, c = 13.12, 3.59991137
""",
            "call": (
                "screened_interaction(chi.copy(), p.copy(), "
                "area, c, beta.copy())"
            ),
            "gold_call": (
                (
                    "_oracle_screened_interaction(chi.copy(), "
                    "p.copy(), area, c, beta.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

chi = np.zeros((2, 2), dtype=complex)
p = np.array([0.5, 2.0])
beta = np.ones(2)
area, c = 4.0, 1.2
""",
            "call": (
                "screened_interaction(chi.copy(), p.copy(), "
                "area, c, beta.copy())"
            ),
            "gold_call": (
                (
                    "_oracle_screened_interaction(chi.copy(), "
                    "p.copy(), area, c, beta.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

b = np.array([[0.3, 0.1j], [0.0, 0.0], [0.2 + 0.1j, -0.3]])
chi = -b @ b.conj().T
p = np.array([0.7, 0.0, 1.3])
beta = np.array([0.6, 1.0, 0.4])
area, c = 6.0, 2.0
""",
            "call": (
                "screened_interaction(chi.copy(), p.copy(), "
                "area, c, beta.copy())"
            ),
            "gold_call": (
                (
                    "_oracle_screened_interaction(chi.copy(), "
                    "p.copy(), area, c, beta.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np

chi = -np.eye(2)
p = np.array([0.0, 1.0])
beta = np.array([1.0, 0.3])
area, c = 3.0, 0.0
""",
            "call": (
                "screened_interaction(chi.copy(), p.copy(), "
                "area, c, beta.copy())"
            ),
            "gold_call": (
                (
                    "_oracle_screened_interaction(chi.copy(), "
                    "p.copy(), area, c, beta.copy())"
                )
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(np.eye(2), np.ones(2), 3.0, 1.0, np.ones(2))
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(screened_interaction)",
            "gold_call": "rejected(_oracle_screened_interaction)",
        },
    ]
