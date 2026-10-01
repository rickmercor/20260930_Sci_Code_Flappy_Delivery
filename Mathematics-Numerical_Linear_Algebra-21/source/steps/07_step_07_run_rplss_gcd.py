"""
Run the complete deterministic randomized projected numerical experiment.

The stable implementation avoids explicit use of the inverse Gram matrix during the iteration. It first prepares a least-squares operator in Step 01, uses that operator to form each current direction, projects s against A P using a least-squares solve, and evaluates the update denominator as ||A p_tilde||^2.

Returns
-------
float, the final squared Euclidean residual
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_rplss_gcd(
    A: np.ndarray,
    x_ref: np.ndarray,
    eta: np.ndarray,
    x0: np.ndarray,
    seed: int = 271828,
    n_iters: int = 4,
    block_size: int = 2,
) -> float:
    """Run the complete deterministic numerical experiment.

    Parameters
    ----------
    A : np.ndarray
        Full-column-rank coefficient matrix of shape (m, n).
    x_ref : np.ndarray
        Reference vector used to construct the right-hand side.
    eta : np.ndarray
        Additive inconsistency vector.
    x0 : np.ndarray
        Initial iterate.
    seed : int
        Seed for the NumPy random number generator.
    n_iters : int
        Number of correction iterations.
    block_size : int
        Number of columns sampled at each iteration.

    Returns
    -------
    float
        Final squared Euclidean residual from the stable float64 recurrence.
        The exact-arithmetic benchmark value is specified by the golden solution.

    Raises
    ------
    ValueError
        If the inputs are invalid, the matrix does not define a valid
        full-column-rank problem, or the requested iteration count would
        exhaust the available correction-space dimension.

    Notes
    -----
    For n=1, one iteration is permitted because no historical correction exists
    before the first update. Otherwise the maximum supported iteration count is n-1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_rplss_gcd(
    A: np.ndarray,
    x_ref: np.ndarray,
    eta: np.ndarray,
    x0: np.ndarray,
    seed: int = 271828,
    n_iters: int = 4,
    block_size: int = 2,
) -> float:
    """Deterministic stable reference orchestrator."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape
    if m < n or n < 1:
        raise ValueError("A must satisfy m >= n >= 1")
    if x_ref.shape != (n,):
        raise ValueError("x_ref has incompatible shape")
    if eta.shape != (m,):
        raise ValueError("eta has incompatible shape")
    if x.shape != (n,):
        raise ValueError("x0 has incompatible shape")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(x_ref)) and np.all(np.isfinite(eta)) and np.all(np.isfinite(x))):
        raise ValueError("inputs must be finite")
    if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1 or block_size > n:
        raise ValueError("invalid block_size")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    max_history_iters = 1 if n == 1 else n - 1
    if n_iters > max_history_iters:
        raise ValueError("n_iters exceeds the available correction-space dimension")

    b = A @ x_ref + eta
    if not np.all(np.isfinite(b)):
        raise ValueError("constructed right-hand side is non-finite")

    # Step 01: prepare the stable least-squares operator consumed by Step 04.
    A_plus = _oracle_prepare_weight_matrix(A)
    if A_plus.shape != (n, m) or not np.all(np.isfinite(A_plus)):
        raise ValueError("invalid least-squares operator")

    blocks = _oracle_generate_column_blocks(
        n_cols=n,
        block_size=int(block_size),
        n_iters=int(n_iters),
        seed=int(seed),
    )
    if blocks.shape != (n_iters, block_size):
        raise ValueError("sampled column blocks have incompatible shape")

    P = np.empty((n, 0), dtype=np.float64)

    for k in range(n_iters):
        tau = blocks[k]
        _, _, s = _oracle_form_residual_sketch(A=A, b=b, x=x, tau=tau)

        projected_direction, delta = _oracle_project_historical_direction(
            A=A,
            A_plus=A_plus,
            P=P,
            s=s,
        )

        p, _ = _oracle_compute_projected_update(
            A=A,
            b=b,
            x=x,
            s=s,
            projected_direction=projected_direction,
            delta=delta,
        )

        x, P = _oracle_update_iterate_state(x=x, P=P, p=p)

    residual = A @ x - b
    if not np.all(np.isfinite(residual)):
        raise ValueError("final residual is non-finite")
    result = float(residual @ residual)
    if not np.isfinite(result):
        raise ValueError("final residual is non-finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""

    return [
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.001, -0.999, 0.501, 1.499],
              [2.0, 3.999, -2.001, 1.002, 3.001],
              [-1.0, -2.002, 1.0005, -0.499, -1.498],
              [0.5, 1.0015, -0.4995, 0.2505, 0.749],
              [3.0, 5.999, -2.998, 1.501, 4.502],
              [-2.0, -4.001, 1.999, -1.0005, -2.999],
              [1.2, 2.401, -1.1995, 0.6004, 1.799],
              [-0.7, -1.399, 0.699, -0.3502, -1.049]], dtype=np.float64)
x_ref = np.array([0.4, -0.8, 1.1, -0.6, 0.9], dtype=np.float64)
eta = np.array([0.12, -0.07, 0.05, 0.09, -0.11, 0.08, -0.04, 0.06], dtype=np.float64)
x0 = np.array([0.15, -0.2, 0.35, -0.1, 0.25], dtype=np.float64)
""",
            "call": "float(run_rplss_gcd(A, x_ref, eta, x0, seed=271828, n_iters=4, block_size=2))",
            "gold_call": "_oracle_run_rplss_gcd(A, x_ref, eta, x0, seed=271828, n_iters=4, block_size=2)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]], dtype=np.float64)
x_ref = np.array([1.0], dtype=np.float64)
eta = np.array([0.1], dtype=np.float64)
x0 = np.array([0.0], dtype=np.float64)
""",
            "call": "float(run_rplss_gcd(A, x_ref, eta, x0, seed=0, n_iters=1, block_size=1))",
            "gold_call": "_oracle_run_rplss_gcd(A, x_ref, eta, x0, seed=0, n_iters=1, block_size=1)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
x_ref = np.array([1.0, 0.5], dtype=np.float64)
eta = np.array([0.1, -0.1], dtype=np.float64)
x0 = np.zeros(2, dtype=np.float64)
def catches_value_error(fn):
    try:
        fn(A, x_ref, eta, x0, seed=0, n_iters=1, block_size=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(run_rplss_gcd)",
            "gold_call": "catches_value_error(_oracle_run_rplss_gcd)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3, dtype=np.float64)
x_ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
eta = np.array([0.1, 0.2, 0.3], dtype=np.float64)
x0 = np.zeros(3, dtype=np.float64)
def catches_value_error(fn):
    try:
        fn(A, x_ref, eta, x0, seed=0, n_iters=3, block_size=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(run_rplss_gcd)",
            "gold_call": "catches_value_error(_oracle_run_rplss_gcd)",
        },
    ]
