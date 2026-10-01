"""
Assemble the full pipeline and return a single entry of the approximated propagator applied to a vector.

Verify that the discrete time-uniform error at the supplied pole interval does not exceed the supplied bound. Reduce that interval to its condenser parameters, construct the extremal configuration of degree n from them, and obtain the residues of the interpolant of exp(-t z) at the evaluation time.

Independently, build the truncated Krylov decomposition of A from b with the given truncation window and condition threshold, and correct its projected matrix by the harmonic rank-one modification.

Form the principal square root S of the corrected projected matrix. For each pole sigma_i, solve the shifted system whose matrix is S minus sigma_i times the identity and whose right-hand side is the first coordinate vector, weight the solutions by the corresponding residues and sum them, apply the approximation basis once to the summed vector, and scale by the Euclidean norm of b. Accumulate the weighted solutions in increasing pole index before applying the basis.

Return the entry of the resulting vector at the requested index, as a native Python float.

The function raises ValueError if any input fails the validity conditions of the steps it invokes, if the index is not an integer in range for the dimension of A, if the error bound is not finite and positive, or if the discrete time-uniform error at the supplied interval exceeds that bound.

The two halves of the method meet at the observation that a shifted resolvent of the square root of a matrix is itself a function of the original matrix. Approximating the propagator by a partial fraction with shared real poles requires, for each pole, the action of the resolvent of $A^{1/2}$ on the starting vector; writing that resolvent as a scalar function of the eigenvalues of $A$ shows it belongs to the Stieltjes class, whose representing measure is nonnegative on the negative real axis. That is precisely the class for which the harmonic projection carries a convergence guarantee on matrices with numerical range in the right half-plane.

The practical consequence is that a single Krylov decomposition suffices for the entire partial fraction. The basis depends only on the matrix and the starting vector, not on the function being applied, so all pole dependence is confined to small systems of the order of the subspace dimension. Applying the basis once to the accumulated small vector, rather than once per pole, both reduces work and avoids repeating the largest operation in the pipeline.

Two independent approximations therefore contribute to the result. The rational family carries an error uniform in time and independent of the matrix, controlled by the degree and the pole interval. The projection carries an error controlled by the subspace dimension, which here is not chosen but is imposed by how quickly the truncated recurrence degrades. Which of the two dominates is a property of the configuration rather than of the method.

Returns
-------
float, the requested entry of the approximated propagator applied to b, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_propagator(A: np.ndarray, b: np.ndarray, n: int, c: float, d: float,
                        times: np.ndarray, z_samples: np.ndarray, t_eval: float,
                        trunc: int, tau: float, index: int,
                        err_bound: float) -> float:
    '''Evaluate one entry of the approximated propagator exp(-t_eval * A**(1/2)) b.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    b : np.ndarray
        Finite starting vector of length N with nonzero norm.
    n : int
        Degree of the shared-pole rational approximation. Positive integer.
    c : float
        Left endpoint of the pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the pole interval. Must be finite with c < d < 0.
    times : np.ndarray
        One-dimensional non-empty array of finite, strictly positive times.
    z_samples : np.ndarray
        One-dimensional non-empty array of finite, nonnegative sample points.
    t_eval : float
        Evaluation time. Must be finite and strictly positive.
    trunc : int
        Truncation window for the Krylov recurrence. Positive integer.
    tau : float
        Basis condition number threshold. Finite and greater than one.
    index : int
        Zero-based entry of the result vector to return.
    err_bound : float
        Upper bound the discrete time-uniform error must satisfy.

    Returns
    -------
    value : float
        The requested entry of the approximated propagator applied to b.

    Raises
    ------
    ValueError
        If any input fails the validity conditions of the steps it invokes, if the
        index is not an integer in range for the dimension of A, if the error bound
        is not finite and positive, or if the discrete time-uniform error at the
        supplied interval exceeds that bound.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as sla


def _oracle_evaluate_propagator(A: np.ndarray, b: np.ndarray, n: int, c: float,
                                d: float, times: np.ndarray, z_samples: np.ndarray,
                                t_eval: float, trunc: int, tau: float,
                                index: int, err_bound: float) -> float:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be one-dimensional of length matching A")
    err_bound = float(err_bound)
    if not np.isfinite(err_bound) or err_bound <= 0.0:
        raise ValueError("err_bound must be finite and positive")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer")
    index = int(index)
    if not (0 <= index < A.shape[0]):
        raise ValueError("index out of range for the dimension of A")

    achieved = _oracle_discrete_uniform_error(c, d, n, times, z_samples)  # noqa: F821
    if not (achieved <= err_bound):
        raise ValueError("interval does not meet the stated error bound")

    params = _oracle_condenser_parameters(c, d)  # noqa: F821
    config = _oracle_zolotarev_poles_nodes(params, n)  # noqa: F821
    poles = config[0]
    nodes = config[1]

    alpha = _oracle_interpolation_residues(poles, nodes, t_eval)  # noqa: F821

    packed = _oracle_truncated_arnoldi_basis(A, b, trunc, tau)  # noqa: F821
    m = packed.shape[1] - 1
    ndim = packed.shape[0] - m - 1
    basis = packed[:ndim, :m]

    h_tilde = _oracle_harmonic_rank_one_update(A, packed)  # noqa: F821

    sqrt_h = np.real(sla.sqrtm(h_tilde))
    e1 = np.zeros(m, dtype=float)
    e1[0] = 1.0
    ident = np.eye(m, dtype=float)

    acc = np.zeros(m, dtype=float)
    for k in range(poles.size):
        acc = acc + alpha[k] * np.linalg.solve(sqrt_h - poles[k] * ident, e1)

    beta = float(np.linalg.norm(b))
    result = beta * (basis @ acc)
    return float(result[index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the full production configuration ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -551.5183157669
d = -10.1118823417
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, "
                     "1.0e4, 321, 1.0e-5)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 1.0, 1, 1.0e4, 321, 1.0e-5)"),
        },
        # --- normal: same configuration, a different output entry ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -551.5183157669
d = -10.1118823417
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, "
                     "1.0e4, 0, 1.0e-5)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 1.0, 1, 1.0e4, 0, 1.0e-5)"),
        },
        # --- normal: earlier evaluation time within the interval ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -551.5183157669
d = -10.1118823417
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 0.1, 1, "
                     "1.0e4, 321, 1.0e-5)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 0.1, 1, 1.0e4, 321, 1.0e-5)"),
        },
        # --- boundary: smaller matrix, low degree, coarse grids ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
n = 8
c = -320.7743354512
d = -3.8926990917
times = np.logspace(-2.0, 0.0, 8)
z_samples = np.concatenate([[0.0], np.logspace(-5.0, 5.0, 400)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, "
                     "1.0e2, 3, 1.0)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 1.0, 1, 1.0e2, 3, 1.0)"),
        },
        # --- edge: tighter condition threshold gives a smaller subspace ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -551.5183157669
d = -10.1118823417
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, "
                     "1.0e3, 321, 1.0e-5)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 1.0, 1, 1.0e3, 321, 1.0e-5)"),
        },
        # --- edge: unrefined interval, admitted only under a looser bound ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -1484.9242404917
d = -14.8492424049
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
""",
            "call": ("evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, "
                     "1.0e4, 321, 1.0e-3)"),
            "gold_call": ("_oracle_evaluate_propagator(A, b, n, c, d, times, "
                          "z_samples, 1.0, 1, 1.0e4, 321, 1.0e-3)"),
        },
        # --- invalid: error bound the given interval cannot meet ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
n = 21
c = -551.5183157669
d = -10.1118823417
times = np.logspace(-2.0, 0.0, 40)
z_samples = np.concatenate([[0.0], np.logspace(-6.0, 6.0, 3000)])
def run_model():
    try:
        evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, 1.0e4,
                            321, 1.0e-9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1,
                                    1.0e4, 321, 1.0e-9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: index out of range ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
n = 8
c = -320.7743354512
d = -3.8926990917
times = np.logspace(-2.0, 0.0, 8)
z_samples = np.concatenate([[0.0], np.logspace(-5.0, 5.0, 400)])
def run_model():
    try:
        evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, 1.0e2, 8, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1,
                                    1.0e2, 8, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: nonpositive error bound ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
n = 8
c = -320.7743354512
d = -3.8926990917
times = np.logspace(-2.0, 0.0, 8)
z_samples = np.concatenate([[0.0], np.logspace(-5.0, 5.0, 400)])
def run_model():
    try:
        evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, 1.0e2, 3, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1,
                                    1.0e2, 3, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: interval ordering violated ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
n = 8
c = -3.8926990917
d = -320.7743354512
times = np.logspace(-2.0, 0.0, 8)
z_samples = np.concatenate([[0.0], np.logspace(-5.0, 5.0, 400)])
def run_model():
    try:
        evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1, 1.0e2, 3, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_propagator(A, b, n, c, d, times, z_samples, 1.0, 1,
                                    1.0e2, 3, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
