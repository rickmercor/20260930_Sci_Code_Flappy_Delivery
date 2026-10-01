"""
Run a fixed number of right-preconditioned LSQR iterations.

Right-preconditioned LSQR applies the packed inverse map inside Golub-Kahan bidiagonalization of the augmented operator. A fixed iteration count makes the result deterministic.

Returns
-------
ndarray of shape (n,): the fixed-iteration approximate minimizer
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def preconditioned_lsqr(
    A: np.ndarray,
    b: np.ndarray,
    pinv_factors: np.ndarray,
    mu: float,
    niter: int,
) -> np.ndarray:
    """Solve the augmented problem using supplied inverse factors.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    b : np.ndarray
        Right-hand side, shape (m,).
    pinv_factors : np.ndarray
        Packed inverse coefficients and right basis, shape (n+1, ell).
    mu : float
        Regularization parameter, mu >= 0.
    niter : int
        Number of Golub-Kahan LSQR iterations, niter >= 1.

    Returns
    -------
    x : np.ndarray
        Approximate minimizer, shape (n,).

    Raises
    ------
    ValueError
        If A is not 2D, if b does not have shape (m,), if pinv_factors is not
        finite with shape (n+1, ell), if the captured right basis does not
        have orthonormal columns, if mu is not a finite number >= 0, or if
        niter is not an integer >= 1.
    """
    return np.zeros(A.shape[1])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _apply_pinv_factors(pinv_factors: np.ndarray, vec: np.ndarray) -> np.ndarray:
    inverse_scale = pinv_factors[0]
    Vhat = pinv_factors[1:]
    return vec + Vhat @ (inverse_scale * (Vhat.T @ vec))


def _lsqr_gk(_matvec, _rmatvec, rhs: np.ndarray, niter: int, dim: int) -> np.ndarray:
    np = __import__("numpy")
    beta = float(np.linalg.norm(rhs))
    if beta == 0.0:
        return np.zeros(dim)
    u = rhs / beta
    tmp = _rmatvec(u)
    alpha = float(np.linalg.norm(tmp))
    if alpha == 0.0:
        return np.zeros(dim)
    v = tmp / alpha
    w = v.copy()
    x = np.zeros(dim)
    phibar = beta
    rhobar = alpha
    for _ in range(niter):
        u = _matvec(v) - alpha * u
        beta = float(np.linalg.norm(u))
        if beta == 0.0:
            break
        u = u / beta
        v = _rmatvec(u) - beta * v
        alpha = float(np.linalg.norm(v))
        if alpha == 0.0:
            rho = np.hypot(rhobar, beta)
            c = rhobar / rho
            phi = c * phibar
            x = x + (phi / rho) * w
            break
        v = v / alpha
        rho = np.hypot(rhobar, beta)
        c = rhobar / rho
        s = beta / rho
        theta = s * alpha
        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
    return x


def _oracle_preconditioned_lsqr(
    A: np.ndarray,
    b: np.ndarray,
    pinv_factors: np.ndarray,
    mu: float,
    niter: int,
) -> np.ndarray:
    np = __import__("numpy")
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    pinv_factors = np.asarray(pinv_factors, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have shape (m,)")
    if (
        pinv_factors.ndim != 2
        or pinv_factors.shape[0] != n + 1
        or pinv_factors.shape[1] < 1
        or pinv_factors.shape[1] > n
        or not np.all(np.isfinite(pinv_factors))
    ):
        raise ValueError("pinv_factors must be finite with shape (n+1, ell)")
    Vhat = pinv_factors[1:]
    gram = Vhat.T @ Vhat
    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=1e-10, atol=1e-10):
        raise ValueError("the captured right basis must have orthonormal columns")
    if not np.isfinite(mu) or float(mu) < 0.0:
        raise ValueError("mu must be a finite number >= 0")
    if not isinstance(niter, (int, np.integer)) or int(niter) < 1:
        raise ValueError("niter must be an integer >= 1")

    mu_f = float(mu)

    def _pinv_map(vec):
        return _apply_pinv_factors(pinv_factors, vec)

    A_mu = np.vstack([A, mu_f * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])

    def _matvec(y):
        return A_mu @ _pinv_map(y)

    def _rmatvec(u):
        return _pinv_map(A_mu.T @ u)

    y = _lsqr_gk(_matvec, _rmatvec, b_aug, int(niter), n)
    return _pinv_map(y)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
left, _ = np.linalg.qr(rng.standard_normal((12, 8)), mode='reduced')
right, _ = np.linalg.qr(rng.standard_normal((8, 8)), mode='reduced')
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
A = left @ np.diag(s) @ right.T
b = rng.standard_normal(12)
index_vector = np.array([2.0, 6.0, 0.0, 6.0])
U = _oracle_cur_core_matrix(A, index_vector)
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
pinv_factors = _oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, 0.05)
mu, niter = 0.05, 2
""",
            "call": "preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
            "gold_call": "_oracle_preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=float)
b = np.array([1.0, 1.0, 0.0], dtype=float)
index_vector = np.array([0.0, 0.0])
U = _oracle_cur_core_matrix(A, index_vector)
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
pinv_factors = _oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, 0.0)
mu, niter = 0.0, 1
""",
            "call": "preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
            "gold_call": "_oracle_preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
b = np.zeros(3)
index_vector = np.array([1.0, 1.0])
U = _oracle_cur_core_matrix(A, index_vector)
sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
pinv_factors = _oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, 0.25)
mu, niter = 0.25, 3
""",
            "call": "preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
            "gold_call": "_oracle_preconditioned_lsqr(A, b, pinv_factors, mu, niter)",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
b = np.ones(3)
pinv_factors = np.vstack([np.zeros(1), np.eye(3)[:, :1]])
def run_model():
    try:
        preconditioned_lsqr(A, b, pinv_factors, 0.1, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_preconditioned_lsqr(A, b, pinv_factors, 0.1, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.eye(3)
b = np.ones(3)
pinv_factors = np.zeros((3, 1))
def run_model():
    try:
        preconditioned_lsqr(A, b, pinv_factors, 0.1, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_preconditioned_lsqr(A, b, pinv_factors, 0.1, 1)
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
