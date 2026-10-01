"""
Compute the entanglement negativity of a bipartite density matrix across a given bipartition, from the trace norm of its partial transpose.



Negativity vanishes on every state with a positive partial transpose, so a nonzero value certifies entanglement, and it is used here to compare the outputs of different causal-order configurations of two channels.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entanglement_negativity(rho: "np.ndarray", dA: int, dB: int) -> float:
    """Compute the negativity of a bipartite state across the A|B partition.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of subsystem A, dA >= 1.
    dB : int
        Dimension of subsystem B, dB >= 1.

    Returns
    -------
    result : float
        The negativity of rho across A|B, as a native Python float.

    Raises
    ------
    ValueError
        If rho does not have shape (dA*dB, dA*dB).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_entanglement_negativity(rho: "np.ndarray", dA: int, dB: int) -> float:
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError("rho must have shape (dA*dB, dA*dB)")
    arr = rho.reshape(dA, dB, dA, dB)
    pt = np.transpose(arr, (0, 3, 2, 1)).reshape(dA * dB, dA * dB)
    w = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    return float((np.sum(np.abs(w)) - 1) / 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: a Bell state, expect negativity 0.5 ---
        {
            "setup": """
import numpy as np
psi = np.array([1,0,0,1],dtype=complex)/np.sqrt(2)
rho = np.outer(psi, psi.conj())
dA, dB = 2, 2
""",
            "call": "entanglement_negativity(rho.copy(), dA, dB)",
            "gold_call": "_oracle_entanglement_negativity(rho.copy(), dA, dB)",
        },
        # --- Boundary: a product state, expect negativity exactly 0 ---
        {
            "setup": """
import numpy as np
psi = np.array([1,0,0,0],dtype=complex)
rho = np.outer(psi, psi.conj())
dA, dB = 2, 2
""",
            "call": "entanglement_negativity(rho.copy(), dA, dB)",
            "gold_call": "_oracle_entanglement_negativity(rho.copy(), dA, dB)",
        },
        # --- Edge: a maximally entangled two-qutrit state ---
        {
            "setup": """
import numpy as np
psi = np.zeros(9, dtype=complex)
for i in range(3):
    psi[i*3+i] = 1/np.sqrt(3)
rho = np.outer(psi, psi.conj())
dA, dB = 3, 3
""",
            "call": "entanglement_negativity(rho.copy(), dA, dB)",
            "gold_call": "_oracle_entanglement_negativity(rho.copy(), dA, dB)",
        },
    ]
