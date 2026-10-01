"""
Evaluate the thermal expectation value of an operator diagonal in site occupations, exp(i sum_i theta_i n_i), from the single-particle correlation matrix of a Gaussian state.

Local twist operators are products of occupation-dependent phases, and because the thermal state is Gaussian their expectation values reduce to a determinant built from the single-particle correlation matrix, which avoids the partition function altogether. Such expectation values can be astronomically small, for example at high temperature on long chains, so the result has to be returned as a logarithm of the modulus together with a phase that respects the branch conventions of real negative values.

Returns
-------
value : np.ndarray -- Float array [ln|<O>|, arg <O>].
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diagonal_twist_expectation(f: "np.ndarray", theta: "np.ndarray") -> "np.ndarray":
    '''Return [ln|<O>|, arg <O>] for O = exp(i sum_i theta_i n_i) in the Gaussian state with correlations f.

    f is the real symmetric correlation matrix <c_i^dag c_j> of a number-non-conserving
    free-fermion thermal state (for example step 02), and n_i = c_i^dag c_i. Return the
    natural logarithm of |<O>| and arg <O> in (-pi, pi]; a computed phase within 1e-9
    of -pi is returned as +pi. ln|<O>| must be finite and accurate to 1e-9 whenever
    <O> != 0, including when |<O>| is far below the smallest positive double.

    Parameters
    ----------
    f : np.ndarray
        Finite real symmetric array of shape (M, M) with eigenvalues in [0, 1].
    theta : np.ndarray
        Finite real array of shape (M,) of phases (radians).

    Returns
    -------
    value : np.ndarray
        Float array [ln|<O>|, arg <O>].

    Raises
    ------
    ValueError
        If f is not a finite real square symmetric matrix, theta has the wrong shape or
        is not finite, or <O> = 0 exactly.
    '''
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_diagonal_twist_expectation(f: "np.ndarray", theta: "np.ndarray") -> "np.ndarray":
    f = np.asarray(f)
    theta = np.asarray(theta)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or np.iscomplexobj(f) or not np.all(np.isfinite(f)):
        raise ValueError("f must be a finite real square matrix")
    if np.max(np.abs(f - f.T)) > 1e-12:
        raise ValueError("f must be symmetric")
    if theta.shape != (f.shape[0],) or np.iscomplexobj(theta) or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a finite real vector matching f")
    matrix = np.eye(f.shape[0]) - f + f * np.exp(1j * theta)[None, :]
    sign, log_abs = np.linalg.slogdet(matrix)
    if sign == 0:
        raise ValueError("<O> vanishes")
    phase = float(np.angle(sign))
    if phase <= -np.pi + 1e-9:
        phase = float(np.pi)
    return np.array([log_abs, phase])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def thermal(n, t1, t2, t3, beta):
    h = np.zeros((2 * n, 2 * n))
    for j in range(n):
        h[2 * j, 2 * j + 1] -= t1
        k = (j + 1) % n
        h[2 * j + 1, 2 * k] -= t2
        h[2 * j, 2 * k + 1] -= t3
    h = h + h.T
    e, v = np.linalg.eigh(h)
    return (v * (0.5 * (1.0 - np.tanh(0.5 * beta * e)))) @ v.T
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
         "call": "diagonal_twist_expectation(thermal(3, 1.0, 0.6, 1.5, 2.0), np.linspace(0.1, 2.3, 6))",
         "gold_call": "_oracle_diagonal_twist_expectation(thermal(3, 1.0, 0.6, 1.5, 2.0), np.linspace(0.1, 2.3, 6))"},
        {"setup": setup,
         "call": "diagonal_twist_expectation(np.diag([0.8, 0.3]), np.array([-np.pi, 0.0]))",
         "gold_call": "_oracle_diagonal_twist_expectation(np.diag([0.8, 0.3]), np.array([-np.pi, 0.0]))"},
        {"setup": setup,
         "call": "diagonal_twist_expectation(np.diag([0.8, 0.3]), np.array([np.pi, 0.0]))",
         "gold_call": "_oracle_diagonal_twist_expectation(np.diag([0.8, 0.3]), np.array([np.pi, 0.0]))"},
        {"setup": setup,
         "call": "diagonal_twist_expectation(0.5 * np.eye(900), np.full(900, 2.9))",
         "gold_call": "_oracle_diagonal_twist_expectation(0.5 * np.eye(900), np.full(900, 2.9))"},
        {"setup": setup,
         "call": "diagonal_twist_expectation(thermal(151, 1.0, 0.6, 1.5, 3.8), 2.0 * np.pi / 151.0 * np.repeat(np.arange(151.0), 2))",
         "gold_call": "_oracle_diagonal_twist_expectation(thermal(151, 1.0, 0.6, 1.5, 3.8), 2.0 * np.pi / 151.0 * np.repeat(np.arange(151.0), 2))"},
        {"setup": setup,
         "call": "diagonal_twist_expectation(thermal(40, 2.0, 0.5, 0.0, 0.02), np.linspace(-3.0, 3.0, 80))",
         "gold_call": "_oracle_diagonal_twist_expectation(thermal(40, 2.0, 0.5, 0.0, 0.02), np.linspace(-3.0, 3.0, 80))"},
        {"setup": guard,
         "call": "raises_value_error(diagonal_twist_expectation, np.eye(2), np.zeros(3))",
         "gold_call": "raises_value_error(_oracle_diagonal_twist_expectation, np.eye(2), np.zeros(3))"},
        {"setup": guard,
         "call": "raises_value_error(diagonal_twist_expectation, np.array([[0.5, 0.1], [0.0, 0.5]]), np.zeros(2))",
         "gold_call": "raises_value_error(_oracle_diagonal_twist_expectation, np.array([[0.5, 0.1], [0.0, 0.5]]), np.zeros(2))"},
        {"setup": guard,
         "call": "raises_value_error(diagonal_twist_expectation, 0.5 * np.eye(2), np.array([np.pi, 0.0]) + np.array([0.0, np.nan]))",
         "gold_call": "raises_value_error(_oracle_diagonal_twist_expectation, 0.5 * np.eye(2), np.array([np.pi, 0.0]) + np.array([0.0, np.nan]))"},
    ]
