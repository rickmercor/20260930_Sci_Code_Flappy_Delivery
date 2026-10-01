"""
Evaluate the two-band Bloch Hamiltonian of the translation-invariant extended SSH chain at a crystal momentum.

With periodic boundary conditions the extended SSH chain is diagonal in crystal momentum, and at each momentum only a two-by-two matrix in the sublattice basis remains. Its off-diagonal element traces a closed loop in the complex plane as the momentum crosses the Brillouin zone, and the winding of that loop, together with the phase conventions of the Bloch basis, determines the topological sector and the geometric phases built on the Bloch eigenvectors.

Returns
-------
parts : np.ndarray -- Float array of shape (2, 2, 2): parts[0] = Re H(k), parts[1] = Im H(k).
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bloch_hamiltonian(k: float, t1: float, t2: float, t3: float) -> "np.ndarray":
    '''Return H(k) = -[[0, h(k)], [conj(h(k)), 0]] with h(k) = t1 + t2 exp(-i k) + t3 exp(+i k).

    This is the Bloch form of the step-01 Hamiltonian with periodic boundary conditions
    in the basis (A, B), with Bloch phases attached to unit-cell positions only (no
    intracell phase).

    Parameters
    ----------
    k : float
        Crystal momentum, finite.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01, finite and >= 0.

    Returns
    -------
    parts : np.ndarray
        Float array of shape (2, 2, 2): parts[0] = Re H(k), parts[1] = Im H(k).

    Raises
    ------
    ValueError
        If k is not finite or a hopping amplitude is negative or not finite.
    '''
    return parts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bloch_hamiltonian(k: float, t1: float, t2: float, t3: float) -> "np.ndarray":
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    hops = np.array([t1, t2, t3], dtype=float)
    if not np.all(np.isfinite(hops)) or np.any(hops < 0.0):
        raise ValueError("hopping amplitudes must be finite and non-negative")
    off = t1 + t2 * np.exp(-1j * k) + t3 * np.exp(1j * k)
    hk = -np.array([[0.0, off], [np.conj(off), 0.0]])
    return np.stack([hk.real, hk.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
"""
    guard = setup + """
def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
"""
    return [
        {"setup": setup,
         "call": "bloch_hamiltonian(0.0, 1.0, 0.6, 1.5)",
         "gold_call": "_oracle_bloch_hamiltonian(0.0, 1.0, 0.6, 1.5)"},
        {"setup": setup,
         "call": "bloch_hamiltonian(np.pi, 1.0, 0.6, 1.5)",
         "gold_call": "_oracle_bloch_hamiltonian(np.pi, 1.0, 0.6, 1.5)"},
        {"setup": setup,
         "call": "bloch_hamiltonian(2.0 * np.pi / 7.0, 0.8, 1.3, 0.0)",
         "gold_call": "_oracle_bloch_hamiltonian(2.0 * np.pi / 7.0, 0.8, 1.3, 0.0)"},
        {"setup": setup,
         "call": "bloch_hamiltonian(-1.1, 2.0, 0.0, 0.4)",
         "gold_call": "_oracle_bloch_hamiltonian(-1.1, 2.0, 0.0, 0.4)"},
        {"setup": setup,
         "call": "bloch_hamiltonian(13.7, 0.3, 0.9, 1.2)",
         "gold_call": "_oracle_bloch_hamiltonian(13.7, 0.3, 0.9, 1.2)"},
        {"setup": guard,
         "call": "raises_value_error(bloch_hamiltonian, np.nan, 1.0, 1.0, 0.0)",
         "gold_call": "raises_value_error(_oracle_bloch_hamiltonian, np.nan, 1.0, 1.0, 0.0)"},
        {"setup": guard,
         "call": "raises_value_error(bloch_hamiltonian, 0.5, 1.0, -1.0, 0.0)",
         "gold_call": "raises_value_error(_oracle_bloch_hamiltonian, 0.5, 1.0, -1.0, 0.0)"},
    ]
