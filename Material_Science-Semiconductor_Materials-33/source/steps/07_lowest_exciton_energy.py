"""
Obtain the lowest Tamm-Dancoff excitation in the direct interaction channel.

The basis consists of vertical valence-to-conduction electron-hole pairs on

an equally weighted momentum mesh. The direct interaction is attractive and

the exchange term is excluded. The result is an excitation energy, not a

binding energy relative to a separately chosen band gap.

Returns
-------
a finite native float, the lowest excitation energy in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lowest_exciton_energy(energies: "np.ndarray", direct: "np.ndarray") -> float:
    """Return the smallest eigenvalue of the direct-channel BSE Hamiltonian.

    Parameters
    ----------
    energies : "np.ndarray"
        Finite real array (K, 2), K >= 1, in eV, ordered valence/conduction
        with a strictly positive direct gap at every momentum.
    direct : "np.ndarray"
        Finite complex array (K, K) in eV, Hermitian to absolute entrywise
        tolerance 1e-10. It includes the mesh factor. Roundoff within that
        tolerance is removed by Hermitian symmetrization.

    Returns
    -------
    energy : float
        Lowest eigenvalue in eV. A degenerate lowest eigenvalue is supported.
        No additional spin factor or momentum weight is applied. Negative
        eigenvalues are returned as mathematical outputs without clipping.

    Raises
    ------
    ValueError
        If a shape, finiteness, real-energy, positive-gap or Hermiticity
        requirement above is violated.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lowest_exciton_energy(
    energies: "np.ndarray", direct: "np.ndarray"
) -> float:
    """Diagonal band differences minus the attractive direct kernel."""
    e = _finite_array(energies, float, "energies")
    d = _finite_array(direct, complex, "direct")
    if e.ndim != 2 or e.shape[1] != 2 or len(e) < 1:
        raise ValueError("energies must have shape (K, 2)")
    if d.shape != (len(e), len(e)) or np.any(e[:, 1] <= e[:, 0]):
        raise ValueError("invalid direct shape or band ordering")
    if not np.allclose(d, d.T.conj(), rtol=0, atol=1e-10):
        raise ValueError("direct must be Hermitian")
    h = np.diag(e[:, 1] - e[:, 0]) - (d + d.T.conj()) / 2
    return float(np.linalg.eigvalsh(h)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """One pair, complex coupling, degeneracy and a zero interaction."""
    return [
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 2.0]])
d = np.array([[0.7]])
""",
            "call": "lowest_exciton_energy(e.copy(), d.copy())",
            "gold_call": (
                "_oracle_lowest_exciton_energy(e.copy(), " "d.copy())"
            ),
        },
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 1.5], [-0.8, 2.1], [-1.2, 1.9]])
d = np.array(
    [[0.6, 0.2j, 0.1], [-0.2j, 0.8, 0.1 + 0.3j], [0.1, 0.1 - 0.3j, 0.5]]
)
""",
            "call": "lowest_exciton_energy(e.copy(), d.copy())",
            "gold_call": (
                "_oracle_lowest_exciton_energy(e.copy(), " "d.copy())"
            ),
        },
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 1.0], [-1.0, 1.0]])
d = np.eye(2) * 3.0
""",
            "call": "lowest_exciton_energy(e.copy(), d.copy())",
            "gold_call": (
                "_oracle_lowest_exciton_energy(e.copy(), " "d.copy())"
            ),
        },
        {
            "setup": """import numpy as np

e = np.array([[-1.0, 1.0], [-0.2, 1.0], [-3.0, 2.0]])
d = np.zeros((3, 3), dtype=complex)
""",
            "call": "lowest_exciton_energy(e.copy(), d.copy())",
            "gold_call": (
                "_oracle_lowest_exciton_energy(e.copy(), " "d.copy())"
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(
            np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(lowest_exciton_energy)",
            "gold_call": "rejected(_oracle_lowest_exciton_energy)",
        },
    ]
