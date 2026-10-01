"""
Solve the positive entropy-regularized spectral stationarity equations at a fixed regularization strength using a safeguarded Newton iteration.

For a fixed regularization parameter $\alpha>0$, the positive spectral amplitudes satisfy the stationarity equations of the entropy-regularized inverse problem.

Let

$$

W=\operatorname{diag}\left(\frac{1}{\sigma_1^2},\ldots,\frac{1}{\sigma_{n_{\mathrm{obs}}}^2}\right).

$$

The gradient of the regularized objective is

$$

\mathbf g(\mathbf a)=K^TW(K\mathbf a-\mathbf d)+\alpha\ln\left(\frac{\mathbf a}{\mathbf m}\right),

$$

where the logarithm is componentwise. Its Hessian is

$$

H(\mathbf a)=K^TWK+\alpha\operatorname{diag}\left(\frac{1}{a_1},\ldots,\frac{1}{a_{n_{\mathrm{basis}}}}\right).

$$

A Newton correction $\mathbf p$ therefore satisfies

$$

H(\mathbf a)\mathbf p=-\mathbf g(\mathbf a).

$$

Because the amplitudes must remain strictly positive, the Newton step is restricted before applying an Armijo backtracking line search to the regularized objective. The positive entropy curvature makes the stationary point unique for $\alpha>0$.

Returns
-------
np.ndarray of shape (n_basis,) containing the unique strictly positive fixed-alpha KKT solution
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_kkt_newton(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    """Solve the positive fixed-alpha KKT equations with safeguarded Newton steps.

    Parameters
    ----------
    kernel : np.ndarray
        Finite two-dimensional array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite one-dimensional correction data with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    alpha : float
        Finite strictly positive regularization strength.
    initial : np.ndarray
        Finite strictly positive initial amplitudes with shape (n_basis,).

    Returns
    -------
    amplitudes : np.ndarray
        Unique strictly positive stationary amplitudes with shape
        (n_basis,).

    Raises
    ------
    ValueError
        If the inputs have inconsistent shapes, contain nonfinite values,
        violate positivity requirements, or alpha is not strictly positive.
    RuntimeError
        If the safeguarded Newton iteration fails to converge.
    """
    return amplitudes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_kkt_newton(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    kernel = np.asarray(kernel, dtype=float)
    correction = np.asarray(correction, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    default = np.asarray(default, dtype=float)
    initial = np.asarray(initial, dtype=float)

    if kernel.ndim != 2 or kernel.shape[0] < 1 or kernel.shape[1] < 1:
        raise ValueError("kernel must be a nonempty two-dimensional array")

    n_obs, n_basis = kernel.shape

    if correction.ndim != 1 or correction.shape != (n_obs,):
        raise ValueError("correction must have shape (n_obs,)")
    if sigma.ndim != 1 or sigma.shape != (n_obs,):
        raise ValueError("sigma must have shape (n_obs,)")
    if default.ndim != 1 or default.shape != (n_basis,):
        raise ValueError("default must have shape (n_basis,)")
    if initial.ndim != 1 or initial.shape != (n_basis,):
        raise ValueError("initial must have shape (n_basis,)")

    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must contain only finite values")
    if not np.all(np.isfinite(correction)):
        raise ValueError("correction must contain only finite values")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must contain only finite values")
    if not np.all(np.isfinite(default)):
        raise ValueError("default must contain only finite values")
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial must contain only finite values")

    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive")
    if np.any(default <= 0.0):
        raise ValueError("default must be strictly positive")
    if np.any(initial <= 0.0):
        raise ValueError("initial must be strictly positive")
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")

    alpha = float(alpha)
    amplitudes = initial.copy()

    inv_var = 1.0 / sigma**2

    gram = kernel.T @ (
        kernel * inv_var[:, None]
    )

    data_scale = 1.0 + np.linalg.norm(
        kernel.T @ (
            correction * inv_var
        ),
        ord=np.inf,
    )

    def _evaluate(values):
        residual = (
            kernel @ values
            - correction
        )

        standardized = (
            residual / sigma
        )

        chi2 = float(
            standardized @ standardized
        )

        entropy = float(
            np.sum(
                values
                * np.log(values / default)
                - values
                + default
            )
        )

        objective = (
            0.5 * chi2
            + alpha * entropy
        )

        gradient = (
            kernel.T @ (
                residual * inv_var
            )
            + alpha
            * np.log(values / default)
        )

        hessian = (
            gram
            + alpha
            * np.diag(
                1.0 / values
            )
        )

        return (
            objective,
            gradient,
            hessian,
        )

    for _ in range(200):
        (
            objective,
            gradient,
            hessian,
        ) = _evaluate(amplitudes)

        if (
            np.linalg.norm(
                gradient,
                ord=np.inf,
            )
            <= 1e-12 * data_scale
        ):
            return amplitudes.astype(float)

        try:
            direction = np.linalg.solve(
                hessian,
                -gradient,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Newton system could not be solved"
            ) from exc

        step = 1.0

        negative = (
            direction < 0.0
        )

        if np.any(negative):
            boundary_step = np.min(
                -0.995
                * amplitudes[negative]
                / direction[negative]
            )

            step = min(
                step,
                float(boundary_step),
            )

        directional_derivative = float(
            gradient @ direction
        )

        accepted = False

        for _ in range(80):
            candidate = (
                amplitudes
                + step * direction
            )

            if np.all(candidate > 0.0):
                (
                    candidate_objective,
                    _,
                    _,
                ) = _evaluate(candidate)

                if (
                    np.isfinite(
                        candidate_objective
                    )
                    and candidate_objective
                    <= objective
                    + 1e-4
                    * step
                    * directional_derivative
                    + 16.0 * np.finfo(float).eps * (1.0 + abs(objective))
                ):
                    amplitudes = candidate
                    accepted = True
                    break

            step *= 0.5

        if not accepted:
            raise RuntimeError(
                "Newton line search failed"
            )

    raise RuntimeError(
        "KKT Newton iteration did not converge"
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
""",
            "call": 'solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
            "gold_call": '_oracle_solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
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
""",
            "call": 'solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
            "gold_call": '_oracle_solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
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
""",
            "call": 'solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
            "gold_call": '_oracle_solve_kkt_newton(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), alpha, initial.copy())',
        },
        {
            "setup": """import numpy as np
kernel = np.eye(2, dtype=float)
correction = np.array([0.1, 0.2], dtype=float)
sigma = np.array([0.1, 0.1], dtype=float)
default = np.array([0.05, 0.10], dtype=float)
alpha = 1.0
initial = np.array([0.05, 0.0], dtype=float)

def run_model():
    try:
        solve_kkt_newton(
            kernel.copy(),
            correction.copy(),
            sigma.copy(),
            default.copy(),
            alpha,
            initial.copy(),
        )
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_solve_kkt_newton(
            kernel.copy(),
            correction.copy(),
            sigma.copy(),
            default.copy(),
            alpha,
            initial.copy(),
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
