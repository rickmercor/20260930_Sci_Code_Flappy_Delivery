"""
Compute the level-k symmetric-extension entanglement measure of a bipartite state from the converged solution of its homogeneous semidefinite program.



Because the domination constraint of the measure's defining program becomes linear once written in the unnormalized extension variable omega' := (1+t)*omega, the trace of the converged solution directly recovers 1+t, so the measure itself is simply Tr(omega') - 1; no outer search over the threshold t is needed.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_Ek(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> float:
    """Compute E_k(rho) from the converged homogeneous-SDP extension operator.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of the first subsystem, dA >= 1.
    dB : int
        Dimension of each of the k extended copies of the second subsystem, dB >= 1.
    k : int
        Number of symmetric extension copies, k >= 1.
    iters : int
        Number of ADMM cycles used to solve the homogeneous SDP.

    Returns
    -------
    result : float
        E_k(rho) = Tr(omega') - 1, where omega' is the converged unnormalized
        extension operator.

    Raises
    ------
    ValueError
        If k < 1 or iters < 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_Ek(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    omega = _oracle_symmetric_extension_sdp_solve(rho, dA, dB, k, iters)
    return float(np.trace(omega).real - 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 2-qubit Werner state at p=0.5, k=1, known E_1=0.25 ---
        {
            "setup": """
import numpy as np
psi_minus = np.array([0,1,-1,0], dtype=complex)/np.sqrt(2)
proj = np.outer(psi_minus, psi_minus.conj())
rho = 0.5*proj + 0.5*np.eye(4)/4
dA, dB, k = 2, 2, 1
iters = 800
""",
            "call": "compute_Ek(rho.copy(), dA, dB, k, iters)",
            "gold_call": "_oracle_compute_Ek(rho.copy(), dA, dB, k, iters)",
            "tol": 0.005,
        },
        # --- Boundary: a separable qubit product state, k=1, expect E_1 close to 0 ---
        {
            "setup": """
import numpy as np
psi0 = np.array([1,0,0,0], dtype=complex)
rho = np.outer(psi0, psi0.conj())
dA, dB, k = 2, 2, 1
iters = 800
""",
            "call": "compute_Ek(rho.copy(), dA, dB, k, iters)",
            "gold_call": "_oracle_compute_Ek(rho.copy(), dA, dB, k, iters)",
            "tol": 0.005,
        },
        # --- Edge: a two-qutrit maximally entangled state, k=2, known E_2=2.0 ---
        {
            "setup": """
import numpy as np
psi = np.zeros(9, dtype=complex)
for i in range(3):
    psi[i*3+i] = 1/np.sqrt(3)
rho = np.outer(psi, psi.conj())
dA, dB, k = 3, 3, 2
iters = 2000
""",
            "call": "compute_Ek(rho.copy(), dA, dB, k, iters)",
            "gold_call": "_oracle_compute_Ek(rho.copy(), dA, dB, k, iters)",
            "tol": 0.005,
        },
    ]
