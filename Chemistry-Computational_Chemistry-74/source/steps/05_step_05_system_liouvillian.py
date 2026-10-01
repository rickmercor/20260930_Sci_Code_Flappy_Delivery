"""
Superoperator of the system alone: coherent evolution under the dipole-gauge Hamiltonian plus Lindblad photon loss from the cavity into a zero-temperature continuum.

Without the baths the density matrix of the electron-photon system obeys d rho / dt = -(i / hbar) [H, rho] + kappa (a rho a^dagger - (1/2) {a^dagger a, rho}), where kappa is the photon loss rate and a acts on the photon factor of the product basis of step 03. Both H and hbar kappa are given in cm^-1; convert them to angular frequencies in rad/ps with 2 pi c, c = 2.99792458e-2 cm/ps.

Density matrices are vectorised row by row: element rho[i, j] of a d x d matrix is entry i * d + j of the vector, and the superoperator L satisfies vec(d rho / dt) = L vec(rho).

Returns
-------
complex numpy.ndarray of shape (d^2, d^2), d = 2 n_photon: the Lindblad system superoperator in ps^-1 for row-major vectorised density matrices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def system_liouvillian(hamiltonian_cm: "np.ndarray", kappa_cm: float, n_photon: int) -> "np.ndarray":
    '''System Liouvillian with Lindblad photon loss, in rad/ps.

    Parameters
    ----------
    hamiltonian_cm : numpy.ndarray
        Hermitian (2 n_photon, 2 n_photon) Hamiltonian in cm^-1 (basis of step 03).
    kappa_cm : float
        Photon loss rate expressed as hbar kappa in cm^-1, non-negative.
    n_photon : int
        Number of photon Fock states in the basis.

    Returns
    -------
    liouvillian : numpy.ndarray
        Complex array of shape (d^2, d^2) with d = 2 n_photon, in ps^-1, acting on
        row-major vectorised density matrices.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer, the Hamiltonian is not a Hermitian
        array of shape (2 n_photon, 2 n_photon) (tolerance 1e-10), or kappa_cm is
        negative or not finite.
    '''
    return liouvillian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _angular_per_wavenumber() -> float:
    """Angular frequency in rad/ps carried by one cm^-1 (2 pi c with c in cm/ps)."""
    import numpy as np
    return 2.0 * np.pi * 2.99792458e-2


def _ladder(n_photon: int) -> np.ndarray:
    """Truncated photon annihilation operator on Fock states 0..n_photon-1."""
    import numpy as np
    return np.diag(np.sqrt(np.arange(1, n_photon, dtype=float)), 1)


def _oracle_system_liouvillian(hamiltonian_cm: "np.ndarray", kappa_cm: float, n_photon: int) -> "np.ndarray":
    import numpy as np
    h = np.asarray(hamiltonian_cm, dtype=complex)
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    if h.ndim != 2 or h.shape != (2 * int(n_photon), 2 * int(n_photon)):
        raise ValueError("hamiltonian must be square with dimension 2 * n_photon")
    if not np.allclose(h, h.conj().T, rtol=0.0, atol=1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    if not np.isfinite(kappa_cm) or kappa_cm < 0.0:
        raise ValueError("cavity loss rate must be non-negative")
    w2c = _angular_per_wavenumber()
    d = h.shape[0]
    eye = np.eye(d)
    hw = h * w2c
    liou = -1j * (np.kron(hw, eye) - np.kron(eye, hw.T))
    a = np.kron(np.eye(2), _ladder(int(n_photon)))
    num = a.T @ a
    k = kappa_cm * w2c
    liou = liou + k * (np.kron(a, a) - 0.5 * np.kron(num, eye) - 0.5 * np.kron(eye, num.T))
    return liou

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a generic Hermitian Hamiltonian on two photon states with loss ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
M = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
H = 40.0 * (M + M.conj().T)
""",
            "call": "system_liouvillian(H.copy(), 10.0, 2)",
            "gold_call": "_oracle_system_liouvillian(H.copy(), 10.0, 2)",
            "tol": 1e-09,
        },
        # --- Boundary: no loss, three photon states ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
M = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
H = 25.0 * (M + M.conj().T)
""",
            "call": "system_liouvillian(H.copy(), 0.0, 3)",
            "gold_call": "_oracle_system_liouvillian(H.copy(), 0.0, 3)",
            "tol": 1e-09,
        },
        # --- Edge: a single photon state, where the loss term cannot act ---
        {
            "setup": """import numpy as np
H = np.array([[5.0, 12.0 - 3.0j], [12.0 + 3.0j, -40.0]])
""",
            "call": "system_liouvillian(H.copy(), 25.0, 1)",
            "gold_call": "_oracle_system_liouvillian(H.copy(), 25.0, 1)",
            "tol": 1e-09,
        },
        # --- Invalid: a non-Hermitian Hamiltonian ---
        {
            "setup": """import numpy as np
H = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
def run_model():
    try:
        system_liouvillian(H.copy(), 10.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_system_liouvillian(H.copy(), 10.0, 2)
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
