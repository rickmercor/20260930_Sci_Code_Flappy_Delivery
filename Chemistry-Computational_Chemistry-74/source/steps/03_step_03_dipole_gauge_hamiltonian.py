"""
System Hamiltonian of a donor-acceptor pair coupled to one cavity mode in the dipole gauge, including the dipole self-energy, on a truncated photon Fock basis.

The electronic space is spanned by the donor D, at energy zero, and the acceptor A, at energy -F where F is the driving force; the two are coupled by the electronic matrix element V_DA. The cavity mode has photon energy hbar omega_c and annihilation operator a, truncated to Fock states 0, ..., n_photon - 1.

The molecular dipole operator is projected on {D, A}. Its projections on the cavity polarization, divided by sqrt(2 hbar omega_c V epsilon_0), are the real dimensionless numbers x_XY with X, Y in {D, A} and x_DA = x_AD; the permanent-dipole (energy-fluctuation) couplings are g_X = i x_XX and the transition coupling is t_DA = i x_DA. In the dipole gauge the light-matter term reads hbar omega_c (a - a^dagger) sum over X, Y of i x_XY |X><Y|, which is the convention used here, and the Hamiltonian also contains the dipole self-energy (mu . epsilon)^2 / (2 V epsilon_0) of the projected dipole operator. Express that term through the same dimensionless couplings.

Product states are ordered with the electronic index outermost: basis index = e * n_photon + m, with e = 0 for D, e = 1 for A and m the photon number.

Returns
-------
complex numpy.ndarray of shape (2 n_photon, 2 n_photon): the dipole-gauge system Hamiltonian in cm^-1, electronic index outermost
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dipole_gauge_hamiltonian(driving_force_cm: float, electronic_coupling_cm: float, cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float, n_photon: int) -> "np.ndarray":
    '''Dipole-gauge Hamiltonian of the donor-acceptor pair plus one cavity mode.

    Parameters
    ----------
    driving_force_cm : float
        Driving force F in cm^-1; E_D = 0 and E_A = -F.
    electronic_coupling_cm : float
        Electronic coupling V_DA between D and A in cm^-1.
    cavity_energy_cm : float
        Photon energy hbar omega_c in cm^-1, positive.
    x_dd, x_aa, x_da : float
        Real dimensionless dipole projections x_DD, x_AA and x_DA = x_AD.
    n_photon : int
        Number of photon Fock states kept (0 .. n_photon - 1), at least 1.

    Returns
    -------
    hamiltonian : numpy.ndarray
        Complex Hermitian array of shape (2 n_photon, 2 n_photon) in cm^-1, basis
        index e * n_photon + m with e = 0 (D), 1 (A) and photon number m.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer, cavity_energy_cm is not positive
        and finite, or any other argument is not finite.
    '''
    return hamiltonian

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


def _oracle_dipole_gauge_hamiltonian(driving_force_cm: float, electronic_coupling_cm: float,
                                     cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                     n_photon: int) -> "np.ndarray":
    import numpy as np
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    if not np.isfinite(cavity_energy_cm) or cavity_energy_cm <= 0.0:
        raise ValueError("cavity photon energy must be positive")
    for v in (driving_force_cm, electronic_coupling_cm, x_dd, x_aa, x_da):
        if not np.isfinite(v):
            raise ValueError("parameters must be finite")
    n = int(n_photon)
    a = _ladder(n)
    ph = np.kron(np.eye(2), a)
    p_d = _electronic(np.diag([1.0, 0.0]), n)
    p_a = _electronic(np.diag([0.0, 1.0]), n)
    sx = _electronic(np.array([[0.0, 1.0], [1.0, 0.0]]), n)
    mu = 1j * (x_dd * p_d + x_aa * p_a + x_da * sx)
    h = (-driving_force_cm * p_a + electronic_coupling_cm * sx
         + cavity_energy_cm * (ph.T @ ph)
         + cavity_energy_cm * (ph - ph.T) @ mu
         - cavity_energy_cm * (mu @ mu))
    return 0.5 * (h + h.conj().T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: opposite permanent dipoles and a transition dipole, three photon states ---
        {
            "setup": """import numpy as np
""",
            "call": "dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.2, -0.1, 0.3, 3)",
            "gold_call": "_oracle_dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.2, -0.1, 0.3, 3)",
            "tol": 1e-09,
        },
        # --- Normal: uphill transfer with parallel permanent dipoles and a negative transition dipole ---
        {
            "setup": """import numpy as np
""",
            "call": "dipole_gauge_hamiltonian(-40.0, 25.0, 150.0, 0.5, 0.35, -0.2, 4)",
            "gold_call": "_oracle_dipole_gauge_hamiltonian(-40.0, 25.0, 150.0, 0.5, 0.35, -0.2, 4)",
            "tol": 1e-09,
        },
        # --- Boundary: all dipole projections zero, the photon decouples ---
        {
            "setup": """import numpy as np
""",
            "call": "dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.0, 0.0, 0.0, 2)",
            "gold_call": "_oracle_dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.0, 0.0, 0.0, 2)",
            "tol": 1e-09,
        },
        # --- Edge: a single photon state, where only the self-energy of the dipoles survives ---
        {
            "setup": """import numpy as np
""",
            "call": "dipole_gauge_hamiltonian(60.0, 10.0, 80.0, 0.4, -0.3, 0.25, 1)",
            "gold_call": "_oracle_dipole_gauge_hamiltonian(60.0, 10.0, 80.0, 0.4, -0.3, 0.25, 1)",
            "tol": 1e-09,
        },
        # --- Invalid: no photon state ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.2, -0.1, 0.3, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_dipole_gauge_hamiltonian(100.0, 30.0, 100.0, 0.2, -0.1, 0.3, 0)
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
