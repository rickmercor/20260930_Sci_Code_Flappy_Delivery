"""
System operators that multiply the two bath coordinates: the acceptor projector for the low-frequency bath and the three-body electron-vibration-photon operator for the intramolecular mode.

The low-frequency bath coordinate Q_1 of step 01 shifts the acceptor energy through Q_1 |A><A|. The intramolecular vibration of step 02, with coordinate Q_2, does not shift the energies; it modulates the charge distribution, so every dimensionless dipole projection of step 03 depends on it as x_XY -> x_XY + Q_2 / (hbar omega_c), with the same unit strength for all four elements XY (any overall strength is absorbed in the reorganization energy of the mode). The shift enters the light-matter coupling term of the dipole-gauge Hamiltonian of step 03 and is neglected in the dipole self-energy.

The system-bath interaction is therefore Q_1 V_1 + Q_2 V_2 with two Hermitian system operators. Return both in the basis of step 03.

Returns
-------
complex numpy.ndarray of shape (2, 2 n_photon, 2 n_photon): the system operators coupled to the low-frequency bath coordinate and to the intramolecular mode coordinate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bath_coupling_operators(n_photon: int) -> "np.ndarray":
    '''System operators V_1 and V_2 that multiply the bath coordinates Q_1 and Q_2.

    Parameters
    ----------
    n_photon : int
        Number of photon Fock states kept, at least 1.

    Returns
    -------
    coupling_ops : numpy.ndarray
        Complex array of shape (2, 2 n_photon, 2 n_photon), dimensionless, basis
        index e * n_photon + m as in step 03: coupling_ops[0] = V_1 (low-frequency
        bath) and coupling_ops[1] = V_2 (intramolecular mode), both Hermitian.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer.
    '''
    return coupling_ops

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ladder(n_photon: int) -> np.ndarray:
    """Truncated photon annihilation operator on Fock states 0..n_photon-1."""
    import numpy as np
    return np.diag(np.sqrt(np.arange(1, n_photon, dtype=float)), 1)


def _electronic(m: np.ndarray, n_photon: int) -> np.ndarray:
    """Embed a 2x2 electronic operator (order D, A) in the electron-photon product space."""
    import numpy as np
    return np.kron(m, np.eye(n_photon))


def _oracle_bath_coupling_operators(n_photon: int) -> "np.ndarray":
    import numpy as np
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    n = int(n_photon)
    ph = np.kron(np.eye(2), _ladder(n))
    field = 1j * (ph - ph.T)
    p_a = _electronic(np.diag([0.0, 1.0]), n).astype(complex)
    q_el = _electronic(np.array([[1.0, 1.0], [1.0, 1.0]]), n)
    return np.array([p_a, field @ q_el])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: three photon states ---
        {
            "setup": """import numpy as np
""",
            "call": "bath_coupling_operators(3)",
            "gold_call": "_oracle_bath_coupling_operators(3)",
            "tol": 1e-10,
        },
        # --- Normal: five photon states ---
        {
            "setup": """import numpy as np
""",
            "call": "bath_coupling_operators(5)",
            "gold_call": "_oracle_bath_coupling_operators(5)",
            "tol": 1e-10,
        },
        # --- Boundary: two photon states, the smallest space in which the field operator acts ---
        {
            "setup": """import numpy as np
""",
            "call": "bath_coupling_operators(2)",
            "gold_call": "_oracle_bath_coupling_operators(2)",
            "tol": 1e-10,
        },
        # --- Edge: a single photon state, where the field operator vanishes ---
        {
            "setup": """import numpy as np
""",
            "call": "bath_coupling_operators(1)",
            "gold_call": "_oracle_bath_coupling_operators(1)",
            "tol": 1e-10,
        },
        # --- Invalid: no photon state ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bath_coupling_operators(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_bath_coupling_operators(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
