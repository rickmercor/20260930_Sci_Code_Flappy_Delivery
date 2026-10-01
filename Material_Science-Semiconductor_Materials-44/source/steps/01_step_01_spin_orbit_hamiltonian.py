"""
Build the spin-orbit part of the hole Hamiltonian in the coupled quasispin and hole-spin space.

The uppermost valence band of cuprous oxide carries a quasispin I = 1 from its orbital part alongside the hole spin S = 1/2, so a hole has six internal components. The spin-orbit interaction splits that space into a twofold level carrying the yellow exciton series and a fourfold level carrying the green one, separated by the spin-orbit parameter. The operator carries no momentum and no position, so it enters every envelope block unchanged. Work in the product basis whose quasispin projections are ordered m_I = +1, 0, -1 and, within each of those, whose hole-spin projections are ordered m_S = +1/2, -1/2, so the quasispin index runs slower. Use the standard ladder-operator phases, which make I_z = diag(1, 0, -1) and S_z = diag(1/2, -1/2) with every component Hermitian. Every later step reuses this basis and ordering. Energies are in meV.

Returns
-------
numpy.ndarray of shape (6, 6), the spin-orbit Hamiltonian in meV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_orbit_hamiltonian(delta: float) -> "np.ndarray":
    '''Spin-orbit part of the cuprous oxide hole Hamiltonian.

    Parameters
    ----------
    delta : float
        Spin-orbit coupling parameter in meV. Must be finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Hermitian array of shape (6, 6) in meV.

    Raises
    ------
    ValueError
        If delta is not finite or is negative.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _spin_operators():
    """Quasispin I (I=1) and hole spin S (S=1/2) in the ordered product basis."""
    root2 = np.sqrt(2.0)
    ix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / root2
    iy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / root2
    iz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]]) / 2.0
    sz = np.diag([1.0, -1.0]).astype(complex) / 2.0
    e3, e2 = np.eye(3), np.eye(2)
    return ([np.kron(m, e2) for m in (ix, iy, iz)],
            [np.kron(e3, m) for m in (sx, sy, sz)])

def _oracle_spin_orbit_hamiltonian(delta: float) -> "np.ndarray":
    d = float(delta)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("delta must be finite and non-negative")
    iop, sop = _spin_operators()
    i_dot_s = sum(iop[a] @ sop[a] for a in range(3))
    return np.real((2.0 / 3.0) * d * (np.eye(6) + i_dot_s))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "spin_orbit_hamiltonian(131.0)",
            "gold_call": "_oracle_spin_orbit_hamiltonian(131.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "spin_orbit_hamiltonian(0.0)",
            "gold_call": "_oracle_spin_orbit_hamiltonian(0.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "spin_orbit_hamiltonian(250.5)",
            "gold_call": "_oracle_spin_orbit_hamiltonian(250.5)",
        },
    ]
