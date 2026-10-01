"""
Compute the local topological indicators of the extended SSH chain at nonzero temperature from centre-of-chain twist expectation values.

In the inversion-symmetric extended SSH chain the Wannier charge centres sit on the cells in the trivial phase, on the intercell bonds in the topological phase with winding +1, and one cell further along the next-nearest-neighbour links in the phase with winding -1. Comparing the magnitudes of two-cell intracell, intercell and next-nearest-neighbour twist expectation values at the centre of the chain turns this picture into two real indicators whose signs identify the phase, and which remain meaningful at nonzero temperature while the purity gap is open.

Returns
-------
indicators : np.ndarray -- Float array [Delta_T3, Delta_I].
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    '''Return [Delta_T3, Delta_I] for the periodic chain at inverse temperature beta.

    Use the periodic step-01 Hamiltonian, the step-02 correlation matrix at beta, the
    step-07 phases with centre cell c = (N - 1) / 2 (N odd), and the step-06 expectation
    values. With |<O>| the modulus of each expectation value,

        Delta_T3 = |<T^intra2_c>| - max(|<T^inter_c>|, |<T^nnn_c>|),
        Delta_I  = |<T^inter_c>| - |<T^nnn_c>|.

    Parameters
    ----------
    n_cells : int
        Odd number of unit cells N >= 3.
    t1, t2, t3 : float
        Hopping amplitudes as in step 01.
    beta : float
        Inverse temperature as in step 02.
    delta_x : float
        Intracell half-separation as in step 07.

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


def _oracle_local_indicators(n_cells: int, t1: float, t2: float, t3: float, beta: float, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 3 or n_cells % 2 == 0:
        raise ValueError("n_cells must be an odd integer >= 3")
    f = _oracle_fermi_correlation_matrix(_oracle_extended_ssh_hamiltonian(n_cells, t1, t2, t3, True), beta)
    centre = (int(n_cells) - 1) // 2
    moduli = {}
    for kind in ("intra2", "inter", "nnn"):
        log_abs, _ = _oracle_diagonal_twist_expectation(f, _oracle_local_twist_phases(n_cells, kind, centre, delta_x))
        moduli[kind] = np.exp(log_abs)
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
         "call": "local_indicators(151, 1.0, 0.6, 1.5, 3.8, 0.25)",
         "gold_call": "_oracle_local_indicators(151, 1.0, 0.6, 1.5, 3.8, 0.25)"},
        {"setup": setup,
         "call": "local_indicators(151, 1.0, 1.5, 0.6, 3.0, 0.25)",
         "gold_call": "_oracle_local_indicators(151, 1.0, 1.5, 0.6, 3.0, 0.25)"},
        {"setup": setup,
         "call": "local_indicators(31, 3.0, 1.0, 0.5, 3.0, 0.25)",
         "gold_call": "_oracle_local_indicators(31, 3.0, 1.0, 0.5, 3.0, 0.25)"},
        {"setup": setup,
         "call": "local_indicators(31, 1.0, 0.6, 1.5, np.inf, 0.25)",
         "gold_call": "_oracle_local_indicators(31, 1.0, 0.6, 1.5, np.inf, 0.25)"},
        {"setup": setup,
         "call": "local_indicators(21, 2.0, 0.7, 0.9, 0.0, 0.1)",
         "gold_call": "_oracle_local_indicators(21, 2.0, 0.7, 0.9, 0.0, 0.1)"},
        {"setup": setup,
         "call": "local_indicators(3, 0.5, 1.0, 0.2, 2.0, 0.25)",
         "gold_call": "_oracle_local_indicators(3, 0.5, 1.0, 0.2, 2.0, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_indicators, 30, 1.0, 0.6, 1.5, 3.8, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_indicators, 30, 1.0, 0.6, 1.5, 3.8, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_indicators, 1, 1.0, 0.6, 1.5, 3.8, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_indicators, 1, 1.0, 0.6, 1.5, 3.8, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_indicators, 31, 1.0, 0.6, 1.5, -1.0, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_indicators, 31, 1.0, 0.6, 1.5, -1.0, 0.25)"},
    ]
