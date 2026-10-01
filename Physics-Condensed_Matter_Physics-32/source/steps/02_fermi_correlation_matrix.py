"""
Compute the single-particle correlation matrix of a free-fermion thermal state at any inverse temperature, including zero temperature.

A thermal state of a quadratic fermionic Hamiltonian is a Gaussian state, fully characterised by its correlation matrix <c_i^dag c_j>, the Fermi function of the single-particle Hamiltonian. Near zero temperature the Fermi function becomes a step, and the natural formula involves exponentials of very large arguments, so a usable implementation must stay exact across the whole range of inverse temperatures. At exactly zero energy the occupation is one half at every temperature, and modes that are zero up to rounding must be treated consistently.

Returns
-------
f : np.ndarray -- Real symmetric float array of shape (M, M).
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fermi_correlation_matrix(h: "np.ndarray", beta: float) -> "np.ndarray":
    '''Return the correlation matrix f = (1 + exp(beta h))^{-1} of the thermal state.

    h is a real symmetric single-particle Hamiltonian with eigen-decomposition
    h = sum_n e_n v_n v_n^T. First replace by exactly zero every eigenvalue with
    |e_n| <= 1e-12 * max(1, max_n |e_n|). Then f = sum_n p_n v_n v_n^T with occupation
    p_n = 1 / (1 + exp(beta e_n)); for beta = inf, p_n is 1, 1/2 or 0 for e_n < 0, = 0 or
    > 0. The occupations must be evaluated without overflow for every beta in [0, inf],
    including beta = 1e300, and must be exactly 1/2 for zeroed eigenvalues at any beta.
    The result does not depend on the choice of eigenvectors within degenerate subspaces.

    Parameters
    ----------
    h : np.ndarray
        Finite real symmetric matrix of shape (M, M), M >= 1.
    beta : float
        Inverse temperature, 0 <= beta <= inf (not NaN).

    Returns
    -------
    f : np.ndarray
        Real symmetric float array of shape (M, M).

    Raises
    ------
    ValueError
        If h is not a finite real square symmetric matrix (tolerance 1e-12 on
        h - h^T), or beta is negative or NaN.
    '''
    return f

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _clean_spectrum(h):
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 1 or np.iscomplexobj(h) or not np.all(np.isfinite(h)):
        raise ValueError("h must be a finite real square matrix")
    if np.max(np.abs(h - h.T)) > 1e-12:
        raise ValueError("h must be symmetric")
    energies, vectors = np.linalg.eigh(h.astype(float))
    cutoff = 1e-12 * max(1.0, float(np.max(np.abs(energies))))
    energies = np.where(np.abs(energies) <= cutoff, 0.0, energies)
    return energies, vectors


def _occupations(energies, beta):
    if np.isnan(beta) or beta < 0.0:
        raise ValueError("beta must be non-negative")
    if np.isinf(beta):
        return np.where(energies < 0.0, 1.0, np.where(energies > 0.0, 0.0, 0.5))
    x = beta * energies
    with np.errstate(over="ignore", invalid="ignore"):
        occ = 0.5 * (1.0 - np.tanh(0.5 * x))
        tail = np.exp(-np.abs(x))
        occ = np.where(x > 20.0, tail / (1.0 + tail), np.where(x < -20.0, 1.0 / (1.0 + tail), occ))
    return np.where(energies == 0.0, 0.5, occ)


def _oracle_fermi_correlation_matrix(h: "np.ndarray", beta: float) -> "np.ndarray":
    energies, vectors = _clean_spectrum(h)
    occ = _occupations(energies, float(beta))
    return (vectors * occ) @ vectors.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def chain(n, t1, t2, t3, periodic=True):
    h = np.zeros((2 * n, 2 * n))
    for j in range(n):
        h[2 * j, 2 * j + 1] -= t1
        if j + 1 < n or periodic:
            k = (j + 1) % n
            h[2 * j + 1, 2 * k] -= t2
            h[2 * j, 2 * k + 1] -= t3
    return h + h.T
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
         "call": "fermi_correlation_matrix(chain(3, 1.0, 0.6, 1.5), 2.0)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(3, 1.0, 0.6, 1.5), 2.0)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(chain(3, 1.0, 0.6, 1.5), 0.0)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(3, 1.0, 0.6, 1.5), 0.0)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(chain(4, 1.0, 1.0, 0.0), np.inf)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(4, 1.0, 1.0, 0.0), np.inf)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(chain(4, 1.0, 1.0, 0.0), 1e300)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(4, 1.0, 1.0, 0.0), 1e300)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(chain(5, 2.0, 0.5, 0.0), 1e5)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(5, 2.0, 0.5, 0.0), 1e5)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(chain(6, 0.3, 1.1, 0.9, False), 37.5)",
         "gold_call": "_oracle_fermi_correlation_matrix(chain(6, 0.3, 1.1, 0.9, False), 37.5)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(np.array([[0.0]]), 3.0)",
         "gold_call": "_oracle_fermi_correlation_matrix(np.array([[0.0]]), 3.0)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(np.diag([1e-12, -1.0, 1.0, 2e-12]), 1e300)",
         "gold_call": "_oracle_fermi_correlation_matrix(np.diag([1e-12, -1.0, 1.0, 2e-12]), 1e300)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(np.diag([3e-12, -3e-12, 1.0, -1.0]), np.inf)",
         "gold_call": "_oracle_fermi_correlation_matrix(np.diag([3e-12, -3e-12, 1.0, -1.0]), np.inf)"},
        {"setup": setup,
         "call": "fermi_correlation_matrix(np.diag([-1e6, 5e-7, 1e6, -2e-6]), 1e8)",
         "gold_call": "_oracle_fermi_correlation_matrix(np.diag([-1e6, 5e-7, 1e6, -2e-6]), 1e8)"},
        {"setup": guard,
         "call": "raises_value_error(fermi_correlation_matrix, chain(3, 1.0, 0.6, 1.5), -1.0)",
         "gold_call": "raises_value_error(_oracle_fermi_correlation_matrix, chain(3, 1.0, 0.6, 1.5), -1.0)"},
        {"setup": guard,
         "call": "raises_value_error(fermi_correlation_matrix, chain(3, 1.0, 0.6, 1.5), np.nan)",
         "gold_call": "raises_value_error(_oracle_fermi_correlation_matrix, chain(3, 1.0, 0.6, 1.5), np.nan)"},
        {"setup": guard,
         "call": "raises_value_error(fermi_correlation_matrix, np.array([[0.0, 1.0], [2.0, 0.0]]), 1.0)",
         "gold_call": "raises_value_error(_oracle_fermi_correlation_matrix, np.array([[0.0, 1.0], [2.0, 0.0]]), 1.0)"},
        {"setup": guard,
         "call": "raises_value_error(fermi_correlation_matrix, np.ones((2, 3)), 1.0)",
         "gold_call": "raises_value_error(_oracle_fermi_correlation_matrix, np.ones((2, 3)), 1.0)"},
    ]
