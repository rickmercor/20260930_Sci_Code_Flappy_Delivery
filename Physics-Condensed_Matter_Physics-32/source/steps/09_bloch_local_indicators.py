"""
Compute the local topological indicators of a very long periodic extended SSH chain without forming any matrix of the size of the chain.

Local twist operators act on at most four orbitals, while the thermal correlation matrix of a chain of tens of thousands of cells has hundreds of millions of entries. For a translation-invariant chain every correlation between two orbitals follows from a single momentum sum over two-by-two Bloch occupation matrices, and the expectation value of an operator that is the identity away from a few orbitals reduces to a determinant on those orbitals alone. Combining the two makes the centre-of-chain indicators available at any chain length, which is what a local measurement on a long chain probes.

Returns
-------
indicators : np.ndarray -- Float array [Delta_T3, Delta_I].
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bloch_local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    '''Return [Delta_T3, Delta_I] exactly as defined in step 08, for chains of up to 20001 cells.

    The chain, thermal state (including the step-02 zero-mode rule and beta = inf), twist
    phases (step 07, centre cell c = (N - 1) / 2) and indicators are exactly those of step 08;
    only the admissible size differs. The result must agree with step 08 to 1e-10 wherever
    step 08 can be evaluated, and must be computable for N up to 20001, where the
    2N-by-2N correlation matrix cannot be formed.

    Parameters
    ----------
    n_cells : int
        Odd number of unit cells N >= 3.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature, 0 <= beta <= inf, as in step 02.
    delta_x : float
        Intracell half-separation, finite, as in step 07.

    Returns
    -------
    indicators : np.ndarray
        Float array [Delta_T3, Delta_I].

    Raises
    ------
    ValueError
        If n_cells is even or smaller than 3, or the inputs are invalid as in steps 01, 02
        and 07.
    '''
    return indicators

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bloch_local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 3 or n_cells % 2 == 0:
        raise ValueError("n_cells must be an odd integer >= 3")
    if np.isnan(beta) or beta < 0.0:
        raise ValueError("beta must be non-negative")
    n = int(n_cells)
    centre = (n - 1) // 2
    ks = 2.0 * np.pi * np.arange(n) / n
    hk = np.array([(lambda parts: parts[0] + 1j * parts[1])(_oracle_bloch_hamiltonian(k, t1, t2, t3)) for k in ks])
    energies, vectors = np.linalg.eigh(hk)
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    occupations = _occupations(energies, float(beta))
    bloch = np.einsum("kai,ki,kbi->kab", vectors, occupations, vectors.conj())
    # <c^dag_{j,a} c_{j+d,b}> = (1/N) sum_k exp(-i k d) [F(k)]_{ab}; the block on the four
    # centre orbitals (A_c, B_c, A_{c+1}, B_{c+1}) is real for this inversion-symmetric chain.
    same = bloch.mean(axis=0)
    forward = np.einsum("k,kab->ab", np.exp(-1j * ks), bloch) / n
    backward = np.einsum("k,kab->ab", np.exp(1j * ks), bloch) / n
    block = np.block([[same, forward], [backward, same]]).real
    block = 0.5 * (block + block.T)
    moduli = {}
    for kind in ("intra2", "inter", "nnn"):
        theta = _oracle_local_twist_phases(n, kind, centre, delta_x)[2 * centre:2 * centre + 4]
        moduli[kind] = np.exp(_oracle_diagonal_twist_expectation(block, theta)[0])
    return np.array([moduli["intra2"] - max(moduli["inter"], moduli["nnn"]), moduli["inter"] - moduli["nnn"]])

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
         "call": "bloch_local_indicators(20001, 1.0, 0.6, 1.5, 9.506626670411208, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(20001, 1.0, 0.6, 1.5, 9.506626670411208, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(151, 1.0, 0.6, 1.5, 3.8, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(151, 1.0, 0.6, 1.5, 3.8, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(20001, 1.0, 1.5, 0.6, 3.0, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(20001, 1.0, 1.5, 0.6, 3.0, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(19999, 3.0, 1.0, 0.5, 3.0, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(19999, 3.0, 1.0, 0.5, 3.0, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(20001, 1.0, 0.6, 1.5, np.inf, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(20001, 1.0, 0.6, 1.5, np.inf, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(4001, 2.0, 0.7, 0.9, 0.0, 0.1)",
         "gold_call": "_oracle_bloch_local_indicators(4001, 2.0, 0.7, 0.9, 0.0, 0.1)"},
        {"setup": setup,
         "call": "bloch_local_indicators(3, 0.5, 1.0, 0.2, 2.0, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(3, 0.5, 1.0, 0.2, 2.0, 0.25)"},
        {"setup": setup,
         "call": "bloch_local_indicators(12345, 0.8, 1.1, 0.3, 40.0, 0.375)",
         "gold_call": "_oracle_bloch_local_indicators(12345, 0.8, 1.1, 0.3, 40.0, 0.375)"},
        {"setup": setup,
         "call": "bloch_local_indicators(9, 1.0, 1.0, 1.0, np.inf, 0.25)",
         "gold_call": "_oracle_bloch_local_indicators(9, 1.0, 1.0, 1.0, np.inf, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(bloch_local_indicators, 20000, 1.0, 0.6, 1.5, 3.8, 0.25)",
         "gold_call": "raises_value_error(_oracle_bloch_local_indicators, 20000, 1.0, 0.6, 1.5, 3.8, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(bloch_local_indicators, 151, 1.0, 0.6, 1.5, -1.0, 0.25)",
         "gold_call": "raises_value_error(_oracle_bloch_local_indicators, 151, 1.0, 0.6, 1.5, -1.0, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(bloch_local_indicators, 151, 1.0, 0.6, 1.5, 3.8, np.nan)",
         "gold_call": "raises_value_error(_oracle_bloch_local_indicators, 151, 1.0, 0.6, 1.5, 3.8, np.nan)"},
    ]
