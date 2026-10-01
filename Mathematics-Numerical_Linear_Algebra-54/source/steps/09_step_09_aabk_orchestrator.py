"""
[ORCHESTRATOR] Run K deterministic iterations of the paper's adaptive, block-averaged, noise-aware Bregman row-action method under the fresh-noise model and report the final squared error against the exact solution.

This step runs Algorithm 1 of the paper end to end: the noise-aware sampling/weight coupling (Step 1), the spectral quantity of the convergence matrix (Step 2), the exact initialization of the auxiliary step-size sequence from the initial Bregman distance to the exact solution (Steps 3-4), the exact error-bound constant gamma of the step-size rule (Step 5), and then, for each iteration, a batch draw with fresh per-draw noise, the averaged direction (Step 6), the adaptive step and auxiliary update (Step 7) and the dual step with primal recovery (Step 8). How these pieces compose within one iteration, and in what order, is not restated here -- consult Algorithm 1 of the paper. The random protocol below is a convention of this task, not of the paper, and is stated in full.

Returns
-------
float — the squared error ||x_K - x_hat||_2^2 after K iterations, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aabk_pipeline(A: "np.ndarray", x_hat: "np.ndarray", sigma: "np.ndarray",
                  lam: float, alpha: float, tau: int, K: int,
                  seed: int) -> float:
    '''Run K iterations of Algorithm 1 of the paper and report the final
    squared error against the exact solution.

    The clean right-hand side is b = A x_hat. The dual iterate starts at
    x*_0 = 0 (so the primal iterate starts at x_0 = grad f*(0)), the
    auxiliary step-size sequence is initialized exactly (using x_hat) as in
    Theorem 2.1, the error-bound constant gamma of the step-size rule takes
    its exact value for this objective (as produced by error_bound_gamma),
    and the objective is f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Random protocol (a convention of this task; use it exactly). A single
    generator rng = np.random.default_rng(seed) supplies all randomness.
    At each iteration k = 0, ..., K-1, in this order:
      1. batch = rng.choice(m, size=tau, replace=True, p=p), where p is the
         (m,) sampling distribution of Step 1 and rows are indexed 0..m-1;
      2. eps = rng.normal(0.0, sigma[batch]), a single call returning the
         tau fresh noise values in draw order;
      3. the fresh noisy value for draw j is b[batch[j]] + eps[j].
    No other random draws are made.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    x_hat : np.ndarray
        (n,) exact solution of the clean system A x = b.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    tau : int
        Batch size, >= 1.
    K : int
        Number of iterations to run, >= 1.
    seed : int
        Seed of the single np.random.default_rng generator.

    Returns
    -------
    error : float
        The squared Euclidean error ||x_K - x_hat||_2^2 after K iterations,
        as a native Python float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if x_hat does not have shape (n,) matching
        A's columns, if sigma does not have shape (m,) matching A's rows or
        has a non-positive entry, if any row of A is exactly the zero
        vector, if x_hat has no nonzero entry, if lam is negative, if alpha
        is not strictly positive, or if tau or K is not a positive integer.
    '''
    return error  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_aabk_pipeline(A: "np.ndarray", x_hat: "np.ndarray", sigma: "np.ndarray",
                          lam: float, alpha: float, tau: int, K: int,
                          seed: int) -> float:
    A = np.asarray(A, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if x_hat.shape != (n,):
        raise ValueError("x_hat must have shape (n,) matching A's columns")
    if sigma.shape != (m,):
        raise ValueError("sigma must have shape (m,) matching A's rows")
    if np.any(sigma <= 0):
        raise ValueError("all entries of sigma must be strictly positive")
    if np.any(np.sum(A ** 2, axis=1) == 0):
        raise ValueError("A must not contain a zero row")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    if not np.any(x_hat):
        raise ValueError("x_hat must have at least one nonzero entry")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (isinstance(tau, (int, np.integer)) and tau >= 1):
        raise ValueError("tau must be a positive integer")
    if not (isinstance(K, (int, np.integer)) and K >= 1):
        raise ValueError("K must be a positive integer")

    b = A @ x_hat
    p, w = _oracle_noise_aware_coupling(A, sigma, alpha)
    sigma_max_T = _oracle_aabk_spectral_bound(A, w, alpha, tau)
    gamma = _oracle_error_bound_gamma(A, x_hat, lam)

    x_star = np.zeros(n)
    x = np.sign(x_star) * np.maximum(np.abs(x_star) - lam, 0.0)
    breg0 = _oracle_sparse_bregman_distance(x_star, x_hat, lam)
    beta = _oracle_aabk_beta_init(A, sigma, p, w, tau, breg0)

    rng = np.random.default_rng(seed)
    for k in range(K):
        batch = rng.choice(m, size=tau, replace=True, p=p)
        eps = rng.normal(0.0, sigma[batch])
        b_noisy = b[batch] + eps
        d = _oracle_aabk_averaged_direction(A, x, w, batch, b_noisy)
        eta, beta = _oracle_aabk_adaptive_step(beta, alpha, gamma, sigma_max_T)
        x_star, x = _oracle_bregman_dual_update(x_star, d, eta, lam)

    diff = x - x_hat
    return float(np.sum(diff ** 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance (6x4, tau = 3, K = 6, seed 3) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x_hat = np.array([2.0, 0.0, -1.0, 0.0])
sigma = np.array([0.5, 0.5, 3.0, 0.5, 3.0, 0.5])
lam = 0.1
alpha = 1.0
tau = 3
K = 6
seed = 3
""",
            "call": "aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
            "gold_call": "_oracle_aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
        },
        # --- Boundary: a single iteration with batch size one (the
        #     method's non-averaged special case) on a small system. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([1.0, -2.0])
sigma = np.array([0.2, 1.0, 0.2])
lam = 0.05
alpha = 1.0
tau = 1
K = 1
seed = 0
""",
            "call": "aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
            "gold_call": "_oracle_aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
        },
        # --- Edge: many iterations with a large batch and lam = 0
        #     (Euclidean objective), where the adaptive step drives the
        #     error well below the noise level. ---
        {
            "setup": """import numpy as np
rng_setup = np.random.default_rng(5)
A = rng_setup.normal(size=(8, 3)) + 3.0 * np.eye(8, 3)
x_hat = np.array([1.0, 0.0, -0.5])
sigma = np.array([0.3, 0.3, 2.0, 0.3, 0.3, 0.3, 2.0, 0.3])
lam = 0.0
alpha = 1.5
tau = 10
K = 40
seed = 9
""",
            "call": "aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
            "gold_call": "_oracle_aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)",
        },
        # --- Invalid: K = 0 -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([1.0, -2.0])
sigma = np.array([0.2, 1.0, 0.2])
lam = 0.05
alpha = 1.0
tau = 1
K = 0
seed = 0
def run_model():
    try:
        aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive noise level -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([1.0, -2.0])
sigma = np.array([0.2, 0.0, 0.2])
lam = 0.05
alpha = 1.0
tau = 2
K = 3
seed = 0
def run_model():
    try:
        aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_pipeline(A, x_hat, sigma, lam, alpha, tau, K, seed)
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
