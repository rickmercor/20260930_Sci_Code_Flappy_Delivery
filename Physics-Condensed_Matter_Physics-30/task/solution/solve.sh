#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from scipy.special import erfc, erfcx


def compute_gaussian_laplace(
    c: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    c = np.asarray(c, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if c.ndim != 1 or c.size == 0:
        raise ValueError("c must be a nonempty one-dimensional array")
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a nonempty one-dimensional array")
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("widths must be a nonempty one-dimensional array")
    if centers.shape != widths.shape:
        raise ValueError("centers and widths must have the same shape")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")
    if not np.all(np.isfinite(widths)):
        raise ValueError("widths must contain only finite values")
    if np.any(c < 0.0):
        raise ValueError("c must be nonnegative")
    if np.any(centers < 0.0):
        raise ValueError("centers must be nonnegative")
    if np.any(widths <= 0.0):
        raise ValueError("widths must be strictly positive")

    c_col = c[:, None]
    mu_row = centers[None, :]
    s_row = widths[None, :]

    argument = (
        c_col * s_row**2 - mu_row
    ) / (np.sqrt(2.0) * s_row)

    mu_grid = np.broadcast_to(mu_row, argument.shape)
    s_grid = np.broadcast_to(s_row, argument.shape)
    c_grid = np.broadcast_to(c_col, argument.shape)

    scaled = np.empty_like(argument)
    positive = argument >= 0.0

    scaled[positive] = (
        np.exp(
            -(mu_grid[positive] ** 2)
            / (2.0 * s_grid[positive] ** 2)
        )
        * erfcx(argument[positive])
    )

    negative = ~positive

    scaled[negative] = (
        np.exp(
            -c_grid[negative] * mu_grid[negative]
            + 0.5 * (c_grid[negative] * s_grid[negative]) ** 2
        )
        * erfc(argument[negative])
    )

    return s_grid * np.sqrt(np.pi / 2.0) * scaled

import numpy as np


def build_imaginary_time_kernel(
    tau: "np.ndarray",
    beta: float,
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    tau = np.asarray(tau, dtype=float)

    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must contain only finite values")
    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    if np.any(tau < 0.0) or np.any(tau > float(beta)):
        raise ValueError("tau values must lie in [0, beta]")

    reflected = float(beta) - tau
    c_all = np.concatenate((tau, reflected))

    transforms = compute_gaussian_laplace(
        c_all,
        centers,
        widths,
    )

    n_tau = tau.size
    forward = transforms[:n_tau]
    backward = transforms[n_tau:]

    return (forward + backward) / np.pi

import numpy as np


def solve_kkt_newton(
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

import numpy as np


def compute_regularization_sensitivity(
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

import numpy as np


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

    amplitudes_prev = solve_kkt_newton(
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
            compute_regularization_sensitivity(
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

        amplitudes_next = solve_kkt_newton(
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
            compute_regularization_sensitivity(
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
            solve_kkt_newton(
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

import numpy as np


def compute_zero_frequency_spectrum(
    amplitudes: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> float:
    """Reference implementation."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError(
            "amplitudes must be a nonempty one-dimensional array"
        )
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError(
            "centers must be a nonempty one-dimensional array"
        )
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError(
            "widths must be a nonempty one-dimensional array"
        )
    if amplitudes.shape != centers.shape or centers.shape != widths.shape:
        raise ValueError(
            "amplitudes, centers, and widths must have the same shape"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must contain only finite values")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")
    if not np.all(np.isfinite(widths)):
        raise ValueError("widths must contain only finite values")
    if np.any(amplitudes < 0.0):
        raise ValueError("amplitudes must be nonnegative")
    if np.any(centers < 0.0):
        raise ValueError("centers must be nonnegative")
    if np.any(widths <= 0.0):
        raise ValueError("widths must be strictly positive")

    zero_weights = np.exp(
        -(centers**2) / (2.0 * widths**2)
    )

    return float(amplitudes @ zero_weights)

import numpy as np


def compute_conductivity_change(
    beta: float,
    harmonic_lambda_zero: float,
    additional_lambda_zero: float,
) -> "np.ndarray":
    """Reference implementation."""
    beta = float(beta)
    harmonic_lambda_zero = float(harmonic_lambda_zero)
    additional_lambda_zero = float(additional_lambda_zero)

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    if (
        not np.isfinite(harmonic_lambda_zero)
        or harmonic_lambda_zero <= 0.0
    ):
        raise ValueError(
            "harmonic_lambda_zero must be finite and strictly positive"
        )
    if (
        not np.isfinite(additional_lambda_zero)
        or additional_lambda_zero < 0.0
    ):
        raise ValueError(
            "additional_lambda_zero must be finite and nonnegative"
        )

    kappa_h = beta**2 * harmonic_lambda_zero

    kappa_total = beta**2 * (
        harmonic_lambda_zero
        + additional_lambda_zero
    )

    delta_percent = 100.0 * (
        kappa_total - kappa_h
    ) / kappa_h

    return np.array(
        [
            kappa_h,
            kappa_total,
            delta_percent,
        ],
        dtype=float,
    )

import numpy as np


def run_total_current_reconstruction(
    tau: "np.ndarray",
    beta: float,
    harmonic_correlation: "np.ndarray",
    total_correlation: "np.ndarray",
    sigma: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
    default: "np.ndarray",
    harmonic_lambda_zero: float,
    target_chi2: float,
    alpha_lower: float,
    alpha_upper: float,
) -> float:
    """Reference implementation."""
    tau = np.asarray(tau, dtype=float)
    harmonic_correlation = np.asarray(
        harmonic_correlation,
        dtype=float,
    )
    total_correlation = np.asarray(
        total_correlation,
        dtype=float,
    )

    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau must be a nonempty one-dimensional array")
    if harmonic_correlation.shape != tau.shape:
        raise ValueError("harmonic_correlation must match tau")
    if total_correlation.shape != tau.shape:
        raise ValueError("total_correlation must match tau")

    correction = (
        total_correlation
        - harmonic_correlation
    )

    kernel = build_imaginary_time_kernel(
        tau,
        beta,
        centers,
        widths,
    )

    fit = trace_discrepancy_continuation(
        kernel,
        correction,
        sigma,
        default,
        target_chi2,
        alpha_lower,
        alpha_upper,
        12,
    )

    amplitudes = fit[2:]

    additional_lambda_zero = (
        compute_zero_frequency_spectrum(
            amplitudes,
            centers,
            widths,
        )
    )

    conductivity = (
        compute_conductivity_change(
            beta,
            harmonic_lambda_zero,
            additional_lambda_zero,
        )
    )

    return float(
        conductivity[2]
    )
SCICODE_GOLD_EOF
