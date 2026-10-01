"""
Compute the logarithm of the grand-canonical partition function of free fermions from their single-particle energies.

For a quadratic fermionic Hamiltonian at zero chemical potential the partition function factorises over single-particle modes, each contributing a factor of one plus a Boltzmann weight. The partition function itself overflows long before the physics becomes extreme, since each negative-energy mode contributes a factor growing exponentially with inverse temperature, so thermodynamic quantities have to be accumulated in logarithmic form.

Returns
-------
log_z : float -- ln Z.
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def log_partition_function(energies: "np.ndarray", beta: float) -> float:
    '''Return ln Z = sum_n ln(1 + exp(-beta e_n)) for single-particle energies e_n.

    First replace by exactly zero every energy with |e_n| <= 1e-12 * max(1, max_n |e_n|).
    The sum must be evaluated without overflow or loss of the O(1) part: the result
    must be accurate to relative 1e-12 whenever it is finite, including beta * |e_n| up
    to 1e300.

    Parameters
    ----------
    energies : np.ndarray
        Nonempty finite real 1-D array of single-particle energies.
    beta : float
        Inverse temperature, finite and >= 0.

    Returns
    -------
    log_z : float
        ln Z.

    Raises
    ------
    ValueError
        If energies is not a nonempty finite real 1-D array, or beta is negative, NaN or
        infinite.
    '''
    return log_z

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_log_partition_function(energies: "np.ndarray", beta: float) -> float:
    energies = np.asarray(energies)
    if energies.ndim != 1 or energies.size == 0 or np.iscomplexobj(energies) or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a nonempty finite real 1-D array")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    energies = energies.astype(float)
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    x = -float(beta) * energies
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    terms = np.where(energies == 0.0, np.log(2.0), terms)
    return float(np.sum(terms))

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
         "call": "log_partition_function(np.array([-2.0, -0.5, 0.5, 2.0]), 1.3)",
         "gold_call": "_oracle_log_partition_function(np.array([-2.0, -0.5, 0.5, 2.0]), 1.3)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([-2.0, -0.5, 0.5, 2.0]), 0.0)",
         "gold_call": "_oracle_log_partition_function(np.array([-2.0, -0.5, 0.5, 2.0]), 0.0)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([-3.0, 3.0, 1e-15, -1e-14]), 1e5)",
         "gold_call": "_oracle_log_partition_function(np.array([-3.0, 3.0, 1e-15, -1e-14]), 1e5)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([-1.0, 1.0]), 1e300)",
         "gold_call": "_oracle_log_partition_function(np.array([-1.0, 1.0]), 1e300)"},
        {"setup": setup,
         "call": "log_partition_function(np.linspace(-3.0, 3.0, 7), 400.0)",
         "gold_call": "_oracle_log_partition_function(np.linspace(-3.0, 3.0, 7), 400.0)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([1e-9, -2e-9]), 1e8)",
         "gold_call": "_oracle_log_partition_function(np.array([1e-9, -2e-9]), 1e8)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([1e-12, -2e-12]), 1e14)",
         "gold_call": "_oracle_log_partition_function(np.array([1e-12, -2e-12]), 1e14)"},
        {"setup": setup,
         "call": "log_partition_function(np.array([5e-7, -2e-6, 3e-7, 1e6]), 1e8)",
         "gold_call": "_oracle_log_partition_function(np.array([5e-7, -2e-6, 3e-7, 1e6]), 1e8)"},
        {"setup": guard,
         "call": "raises_value_error(log_partition_function, np.array([1.0, -1.0]), -0.1)",
         "gold_call": "raises_value_error(_oracle_log_partition_function, np.array([1.0, -1.0]), -0.1)"},
        {"setup": guard,
         "call": "raises_value_error(log_partition_function, np.array([1.0, -1.0]), np.inf)",
         "gold_call": "raises_value_error(_oracle_log_partition_function, np.array([1.0, -1.0]), np.inf)"},
        {"setup": guard,
         "call": "raises_value_error(log_partition_function, np.array([]), 1.0)",
         "gold_call": "raises_value_error(_oracle_log_partition_function, np.array([]), 1.0)"},
        {"setup": guard,
         "call": "raises_value_error(log_partition_function, np.array([1.0, np.nan]), 1.0)",
         "gold_call": "raises_value_error(_oracle_log_partition_function, np.array([1.0, np.nan]), 1.0)"},
    ]
