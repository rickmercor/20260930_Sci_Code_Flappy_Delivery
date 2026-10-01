"""
Evolve the deterministic averaged state of an open quantum system forward in time by exact propagation of its (time-independent) Lindblad generator, via the matrix exponential of the generator's vectorized superoperator.



This averaged, noise-free evolution is the building block that a maximum-contrast parameter estimator for continuously monitored open quantum systems evaluates repeatedly at trial parameter values. Using the exact propagator, rather than a finite-step discretization, keeps every evaluated state exactly trace preserving and positive semidefinite, and matches how the deterministic averaged dynamics entering the contrast function are evaluated.

Returns
-------
return traj
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def averaged_lindblad_evolution(H: "np.ndarray", Ls: "np.ndarray", rho0: "np.ndarray", T: float, n: int) -> "np.ndarray":
    """Evolve the averaged Lindblad master equation forward by exact propagation.

    Parameters
    ----------
    H : np.ndarray
        (d, d) complex Hermitian Hamiltonian.
    Ls : np.ndarray
        (r, d, d) complex array of r jump operators.
    rho0 : np.ndarray
        (d, d) complex Hermitian, trace-1 initial state.
    T : float
        Total evolution time, T > 0.
    n : int
        Number of equally spaced time points at which to report the state, n >= 1.

    Returns
    -------
    traj : np.ndarray
        (n+1, d, d) complex array, the averaged state at t_0, ..., t_n, obtained by
        repeatedly applying the exact propagator exp((T/n) * L) to rho0.

    Raises
    ------
    ValueError
        If T <= 0 or n < 1.
    """
    return traj

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _lindblad_superop(H: "np.ndarray", Ls: "list") -> "np.ndarray":
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(), L) - 0.5 * (np.kron(I, L.conj().T @ L) + np.kron((L.conj().T @ L).T, I))
    return Lsup


def _apply_superop(Esup: "np.ndarray", rho: "np.ndarray") -> "np.ndarray":
    d = rho.shape[0]
    vecrho = rho.reshape(-1, 1, order="F")
    out = Esup @ vecrho
    return out.reshape(d, d, order="F")


def _oracle_averaged_lindblad_evolution(H: "np.ndarray", Ls: "np.ndarray", rho0: "np.ndarray", T: float, n: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    d = H.shape[0]
    dt = T / n
    Lsup = _lindblad_superop(H, list(Ls))
    Esup = expm(Lsup * dt)
    rho = np.array(rho0, dtype=complex)
    traj = [rho.copy()]
    for _ in range(n):
        rho = _apply_superop(Esup, rho)
        rho = (rho + rho.conj().T) / 2
        traj.append(rho.copy())
    return np.array(traj)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: qubit, single jump operator, a few steps ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(1)
A = rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))
H = (A + A.conj().T)/2
L0 = rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))
Ls = np.array([L0])
rho0 = np.array([[1.0,0.0],[0.0,0.0]], dtype=complex)
T = 0.5
n = 5
""",
            "call": "averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
            "gold_call": "_oracle_averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
        },
        # --- Boundary: no jump operators (purely unitary drift), n=1 ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(2)
A = rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))
H = (A + A.conj().T)/2
Ls = np.zeros((0,2,2), dtype=complex)
rho0 = np.array([[0.5,0.0],[0.0,0.5]], dtype=complex)
T = 0.2
n = 1
""",
            "call": "averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
            "gold_call": "_oracle_averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
        },
        # --- Edge: the actual qutrit generators and settings used in the task instance ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(101)
A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
H0 = (A + A.conj().T)/2
L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
alpha_hat, beta_hat = 1.89302208806816, 0.7232072266302545
H = alpha_hat*H0
Ls = np.array([beta_hat*L0])
rho0 = np.zeros((3,3), dtype=complex)
rho0[0,0] = 1.0
T = 1.0
n = 40
""",
            "call": "averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
            "gold_call": "_oracle_averaged_lindblad_evolution(H.copy(), Ls.copy(), rho0.copy(), T, n)",
        },
    ]
