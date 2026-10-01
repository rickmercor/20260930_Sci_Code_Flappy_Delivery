"""
Compute the implicit derivative of the regularized spectral amplitudes and chi-square discrepancy with respect to the logarithm of the regularization strength.

The discrepancy-matched regularization parameter can be located more efficiently by differentiating the fixed-$\alpha$ KKT equations along the regularized solution branch.

Let

$$

u=\log\alpha.

$$

At a stationary solution $\mathbf a_\alpha$,

$$

K^TW(K\mathbf a_\alpha-\mathbf d)+\alpha\ln\left(\frac{\mathbf a_\alpha}{\mathbf m}\right)=0.

$$

Differentiating with respect to $u$ gives

$$

H\frac{d\mathbf a_\alpha}{du}=-\alpha\ln\left(\frac{\mathbf a_\alpha}{\mathbf m}\right),

$$

where

$$

H=K^TWK+\alpha\operatorname{diag}\left(\frac{1}{a_1},\ldots,\frac{1}{a_{n_{\mathrm{basis}}}}\right).

$$

The discrepancy derivative is then

$$

\frac{d\chi^2}{du}=2(K\mathbf a_\alpha-\mathbf d)^TWK\frac{d\mathbf a_\alpha}{du}.

$$

Returns
-------
np.ndarray of length n_basis + 1 containing the amplitude continuation tangent followed by dchi2/dlog(alpha)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_regularization_sensitivity(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    amplitudes: "np.ndarray",
) -> "np.ndarray":
    """Compute the logarithmic-alpha continuation tangent.

    Parameters
    ----------
    kernel : np.ndarray
        Finite array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite correction vector with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    alpha : float
        Finite strictly positive regularization strength.
    amplitudes : np.ndarray
        Finite strictly positive stationary amplitudes with shape
        (n_basis,).

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length n_basis + 1. The first n_basis
        entries contain d amplitudes / d log(alpha), and the last entry
        contains d chi-square / d log(alpha).

    Raises
    ------
    ValueError
        If dimensions or shapes are inconsistent, if required values are
        nonfinite, or if positivity conditions are violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_regularization_sensitivity(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    amplitudes: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    kernel = np.asarray(kernel, dtype=float)
    correction = np.asarray(correction, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    default = np.asarray(default, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)

    if kernel.ndim != 2 or kernel.shape[0] < 1 or kernel.shape[1] < 1:
        raise ValueError("kernel must be a nonempty two-dimensional array")

    n_obs, n_basis = kernel.shape

    if correction.shape != (n_obs,):
        raise ValueError("correction must have shape (n_obs,)")
    if sigma.shape != (n_obs,):
        raise ValueError("sigma must have shape (n_obs,)")
    if default.shape != (n_basis,):
        raise ValueError("default must have shape (n_basis,)")
    if amplitudes.shape != (n_basis,):
        raise ValueError("amplitudes must have shape (n_basis,)")

    arrays = (
        kernel,
        correction,
        sigma,
        default,
        amplitudes,
    )

    if not all(
        np.all(np.isfinite(values))
        for values in arrays
    ):
        raise ValueError(
            "array inputs must contain only finite values"
        )

    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive")
    if np.any(default <= 0.0):
        raise ValueError("default must be strictly positive")
    if np.any(amplitudes <= 0.0):
        raise ValueError("amplitudes must be strictly positive")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")

    alpha = float(alpha)
    inv_var = 1.0 / sigma**2

    hessian = (
        kernel.T @ (
            kernel * inv_var[:, None]
        )
        + alpha
        * np.diag(1.0 / amplitudes)
    )

    rhs = (
        -alpha
        * np.log(amplitudes / default)
    )

    try:
        tangent = np.linalg.solve(
            hessian,
            rhs,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "sensitivity system could not be solved"
        ) from exc

    residual = (
        kernel @ amplitudes
        - correction
    )

    dchi2 = 2.0 * float(
        (residual * inv_var)
        @ (kernel @ tangent)
    )

    return np.concatenate(
        (
            tangent.astype(float),
            np.array([dchi2], dtype=float),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return comparisons with independent candidate/reference inputs."""
    return [
        {
            "setup": """import numpy as np
kernel = np.array([
    [1.0, 0.2],
    [0.3, 1.1],
    [0.7, 0.4],
], dtype=float)
correction = np.array([0.8, 0.6, 0.5], dtype=float)
sigma = np.array([0.10, 0.15, 0.12], dtype=float)
default = np.array([0.20, 0.30], dtype=float)
alpha = 10.0
initial = np.array([0.05, 0.80], dtype=float)
amplitudes = _oracle_solve_kkt_newton(
    kernel,
    correction,
    sigma,
    default,
    alpha,
    initial,
)
""",
            "call": 'compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
            "gold_call": '_oracle_compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
        },
        {
            "setup": """import numpy as np
kernel = np.array([
    [1.0, 2.0, 0.5],
], dtype=float)
correction = np.array([0.75], dtype=float)
sigma = np.array([0.20], dtype=float)
default = np.array([0.05, 0.10, 0.15], dtype=float)
alpha = 0.2
initial = np.array([0.50, 0.02, 0.70], dtype=float)
amplitudes = _oracle_solve_kkt_newton(
    kernel,
    correction,
    sigma,
    default,
    alpha,
    initial,
)
""",
            "call": 'compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
            "gold_call": '_oracle_compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
        },
        {
            "setup": """import numpy as np
kernel = np.array([
    [1.0, 1.00010, 0.2],
    [0.4, 0.40005, 1.0],
    [0.8, 0.80008, 0.5],
    [0.2, 0.20002, 0.7],
], dtype=float)
correction = np.array([0.45, 0.32, 0.51, 0.28], dtype=float)
sigma = np.array([0.001, 0.02, 0.20, 0.50], dtype=float)
default = np.array([0.001, 0.05, 0.50], dtype=float)
alpha = 50.0
initial = np.array([0.20, 0.001, 0.05], dtype=float)
amplitudes = _oracle_solve_kkt_newton(
    kernel,
    correction,
    sigma,
    default,
    alpha,
    initial,
)
""",
            "call": 'compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
            "gold_call": '_oracle_compute_regularization_sensitivity(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, amplitudes.copy())',
        },
    ]
