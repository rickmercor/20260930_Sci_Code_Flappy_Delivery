"""
Trace the entropy-regularized solution branch in log regularization strength and locate the spectrum satisfying a prescribed chi-square discrepancy using predictor-corrector continuation and safeguarded Newton refinement.

Rather than solving unrelated optimization problems for arbitrary regularization strengths, the regularized spectrum can be followed continuously as a function of

$$

u=\log\alpha.

$$

At a continuation point $u_k$, the implicit sensitivity provides the local tangent

$$

\frac{d\mathbf a}{du}.

$$

For a step $\Delta u$, a first-order predictor is

$$

\mathbf a_{\mathrm{pred}}=\mathbf a_k+\Delta u\frac{d\mathbf a}{du}.

$$

The predicted amplitudes are corrected by solving the fixed-$\alpha$ KKT equations at the new regularization strength.

When the discrepancy residual

$$

f(u)=\chi^2(\mathbf a_{e^u})-\chi^2_{\mathrm{target}}

$$

changes sign, the root is refined using the derivative $d\chi^2/du$. Newton updates that leave the current bracket are replaced by a bisection step.

Returns
-------
np.ndarray of length n_basis + 2 containing [matched alpha, achieved chi-square, fitted amplitudes]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trace_discrepancy_continuation(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    target_chi2: float,
    alpha_lower: float,
    alpha_upper: float,
    n_steps: int,
) -> "np.ndarray":
    """Locate the discrepancy-matched spectrum by log-alpha continuation.

    Parameters
    ----------
    kernel : np.ndarray
        Finite array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite correction data with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    target_chi2 : float
        Finite nonnegative discrepancy target.
    alpha_lower : float
        Finite strictly positive lower continuation bound.
    alpha_upper : float
        Finite strictly positive upper continuation bound greater than
        alpha_lower.
    n_steps : int
        Number of positive log-alpha continuation intervals. Must be at
        least 2.

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length n_basis + 2 containing the matched
        alpha, achieved chi-square, and fitted amplitudes.

    Raises
    ------
    ValueError
        If scalar conditions are violated or the continuation interval
        does not bracket the requested discrepancy.
    RuntimeError
        If continuation or safeguarded root refinement fails.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_trace_discrepancy_continuation(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    target_chi2: float,
    alpha_lower: float,
    alpha_upper: float,
    n_steps: int,
) -> "np.ndarray":
    """Reference implementation."""
    kernel = np.asarray(kernel, dtype=float)
    correction = np.asarray(correction, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    default = np.asarray(default, dtype=float)

    if kernel.ndim != 2 or kernel.shape[0] < 1 or kernel.shape[1] < 1:
        raise ValueError("kernel must be a nonempty two-dimensional array")

    n_obs, n_basis = kernel.shape

    if correction.shape != (n_obs,):
        raise ValueError("correction must have shape (n_obs,)")
    if sigma.shape != (n_obs,):
        raise ValueError("sigma must have shape (n_obs,)")
    if default.shape != (n_basis,):
        raise ValueError("default must have shape (n_basis,)")

    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must contain only finite values")
    if not np.all(np.isfinite(correction)):
        raise ValueError("correction must contain only finite values")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must contain only finite values")
    if not np.all(np.isfinite(default)):
        raise ValueError("default must contain only finite values")

    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive")
    if np.any(default <= 0.0):
        raise ValueError("default must be strictly positive")

    if not np.isfinite(target_chi2) or float(target_chi2) < 0.0:
        raise ValueError("target_chi2 must be finite and nonnegative")
    if not np.isfinite(alpha_lower) or float(alpha_lower) <= 0.0:
        raise ValueError("alpha_lower must be finite and strictly positive")
    if not np.isfinite(alpha_upper) or float(alpha_upper) <= 0.0:
        raise ValueError("alpha_upper must be finite and strictly positive")
    if float(alpha_upper) <= float(alpha_lower):
        raise ValueError("alpha_upper must exceed alpha_lower")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError("n_steps must be an integer")
    if int(n_steps) < 2:
        raise ValueError("n_steps must be at least 2")

    target_chi2 = float(target_chi2)
    alpha_lower = float(alpha_lower)
    alpha_upper = float(alpha_upper)
    n_steps = int(n_steps)

    def _discrepancy(amplitudes):
        standardized = (
            kernel @ amplitudes
            - correction
        ) / sigma

        return float(
            standardized @ standardized
        )

    log_grid = np.linspace(
        np.log(alpha_lower),
        np.log(alpha_upper),
        n_steps + 1,
    )

    alpha_prev = float(
        np.exp(log_grid[0])
    )

    amplitudes_prev = _oracle_solve_kkt_newton(
        kernel,
        correction,
        sigma,
        default,
        alpha_prev,
        default.copy(),
    )

    chi2_prev = _discrepancy(
        amplitudes_prev
    )

    f_prev = chi2_prev - target_chi2

    root_tolerance = (
        1e-11
        * (1.0 + target_chi2)
    )

    if abs(f_prev) <= root_tolerance:
        return np.concatenate(
            (
                np.array(
                    [alpha_prev, chi2_prev],
                    dtype=float,
                ),
                amplitudes_prev,
            )
        )

    bracket = None

    for index in range(1, log_grid.size):
        u_prev = float(
            log_grid[index - 1]
        )

        u_next = float(
            log_grid[index]
        )

        sensitivity = (
            _oracle_compute_regularization_sensitivity(
                kernel,
                correction,
                sigma,
                default,
                alpha_prev,
                amplitudes_prev,
            )
        )

        tangent = sensitivity[:-1]

        predicted = (
            amplitudes_prev
            + (u_next - u_prev)
            * tangent
        )

        positive_floor = np.maximum(
            1e-14,
            1e-12 * default,
        )

        predicted = np.maximum(
            predicted,
            positive_floor,
        )

        alpha_next = float(
            np.exp(u_next)
        )

        amplitudes_next = _oracle_solve_kkt_newton(
            kernel,
            correction,
            sigma,
            default,
            alpha_next,
            predicted,
        )

        chi2_next = _discrepancy(
            amplitudes_next
        )

        f_next = (
            chi2_next
            - target_chi2
        )

        if f_prev * f_next <= 0.0:
            bracket = (
                u_prev,
                amplitudes_prev.copy(),
                f_prev,
                chi2_prev,
                u_next,
                amplitudes_next.copy(),
                f_next,
                chi2_next,
            )
            break

        alpha_prev = alpha_next
        amplitudes_prev = amplitudes_next
        chi2_prev = chi2_next
        f_prev = f_next

    if bracket is None:
        raise ValueError(
            "continuation interval does not bracket the target discrepancy"
        )

    (
        u_low,
        amplitudes_low,
        f_low,
        chi2_low,
        u_high,
        amplitudes_high,
        f_high,
        chi2_high,
    ) = bracket

    if abs(f_low) <= abs(f_high):
        u_current = u_low
        amplitudes_current = amplitudes_low
        f_current = f_low
        chi2_current = chi2_low
    else:
        u_current = u_high
        amplitudes_current = amplitudes_high
        f_current = f_high
        chi2_current = chi2_high

    for _ in range(80):
        if abs(f_current) <= root_tolerance:
            alpha_star = float(
                np.exp(u_current)
            )

            return np.concatenate(
                (
                    np.array(
                        [
                            alpha_star,
                            chi2_current,
                        ],
                        dtype=float,
                    ),
                    amplitudes_current,
                )
            )

        alpha_current = float(
            np.exp(u_current)
        )

        sensitivity = (
            _oracle_compute_regularization_sensitivity(
                kernel,
                correction,
                sigma,
                default,
                alpha_current,
                amplitudes_current,
            )
        )

        tangent = sensitivity[:-1]
        dchi2_du = float(
            sensitivity[-1]
        )

        if (
            np.isfinite(dchi2_du)
            and dchi2_du != 0.0
        ):
            u_trial = (
                u_current
                - f_current
                / dchi2_du
            )
        else:
            u_trial = np.nan

        margin = (
            1e-12
            * (
                1.0
                + abs(u_low)
                + abs(u_high)
            )
        )

        if (
            not np.isfinite(u_trial)
            or u_trial <= u_low + margin
            or u_trial >= u_high - margin
        ):
            u_trial = 0.5 * (
                u_low + u_high
            )

        predicted = (
            amplitudes_current
            + (u_trial - u_current)
            * tangent
        )

        positive_floor = np.maximum(
            1e-14,
            1e-12 * default,
        )

        predicted = np.maximum(
            predicted,
            positive_floor,
        )

        alpha_trial = float(
            np.exp(u_trial)
        )

        amplitudes_trial = (
            _oracle_solve_kkt_newton(
                kernel,
                correction,
                sigma,
                default,
                alpha_trial,
                predicted,
            )
        )

        chi2_trial = _discrepancy(
            amplitudes_trial
        )

        f_trial = (
            chi2_trial
            - target_chi2
        )

        if f_low * f_trial <= 0.0:
            u_high = u_trial
            amplitudes_high = amplitudes_trial
            f_high = f_trial
            chi2_high = chi2_trial
        else:
            u_low = u_trial
            amplitudes_low = amplitudes_trial
            f_low = f_trial
            chi2_low = chi2_trial

        u_current = u_trial
        amplitudes_current = amplitudes_trial
        f_current = f_trial
        chi2_current = chi2_trial

    raise RuntimeError(
        "discrepancy continuation did not converge"
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
target_chi2 = 6.0
alpha_lower = 20.0
alpha_upper = 40.0
n_steps = 4
""",
            "call": 'trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
            "gold_call": '_oracle_trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
        },
        {
            "setup": """import numpy as np
kernel = np.array([
    [0.222184210071, 0.255572664079, 0.268468884142, 0.301716368438, 0.351736576506, 0.412112889181, 0.484911098611],
    [0.220408130444, 0.247148253103, 0.246093600323, 0.254453181961, 0.267904417250, 0.277585941142, 0.285695698114],
    [0.218444635231, 0.237946394654, 0.222285200778, 0.206656114319, 0.189707694348, 0.166273005092, 0.143219481772],
    [0.216780365784, 0.230238427456, 0.202854046148, 0.169613963408, 0.134062930081, 0.096871778226, 0.068008676007],
    [0.215598686746, 0.224816503153, 0.189467630537, 0.145154571243, 0.099892767557, 0.059013117256, 0.032935776825],
    [0.215102825543, 0.222553925444, 0.183950657911, 0.135330963812, 0.086780361087, 0.045575287163, 0.021778687162],
], dtype=float)
correction = np.array([
    0.069244048566,
    0.048820260964,
    0.034425796087,
    0.023371336138,
    0.018713928683,
    0.015610293767,
], dtype=float)
sigma = np.array([
    0.0014,
    0.0012,
    0.0010,
    0.0009,
    0.0008,
    0.0008,
], dtype=float)
default = np.array([
    0.003,
    0.006,
    0.012,
    0.020,
    0.030,
    0.025,
    0.015,
], dtype=float)
target_chi2 = 6.0
alpha_lower = 1100.0
alpha_upper = 1200.0
n_steps = 8
""",
            "call": 'trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
            "gold_call": '_oracle_trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
        },
        {
            "setup": """import numpy as np
kernel = np.array([
    [1.0, 0.2],
    [0.3, 1.1],
], dtype=float)
correction = np.array([0.4, 0.3], dtype=float)
sigma = np.array([0.1, 0.1], dtype=float)
default = np.array([0.1, 0.2], dtype=float)
target_chi2 = 1000.0
alpha_lower = 1.0
alpha_upper = 2.0
n_steps = 4

def run_model():
    try:
        trace_discrepancy_continuation(
            kernel.copy(),
            correction.copy(),
            sigma.copy(),
            default.copy(),
            target_chi2,
            alpha_lower,
            alpha_upper,
            n_steps,
        )
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_trace_discrepancy_continuation(
            kernel.copy(),
            correction.copy(),
            sigma.copy(),
            default.copy(),
            target_chi2,
            alpha_lower,
            alpha_upper,
            n_steps,
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            "setup": """import numpy as np
kernel = np.array([[1.0]])
correction = np.array([2.0])
sigma = np.array([1.0])
default = np.array([np.exp(-1.0)])
target_chi2 = 1.0
alpha_lower = 1.0
alpha_upper = 2.0
n_steps = 2
""",
            "call": 'trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
            "gold_call": '_oracle_trace_discrepancy_continuation(kernel.copy(), correction.copy(), sigma.copy(), default.copy(), target_chi2, alpha_lower, alpha_upper, n_steps)',
        },
    ]
