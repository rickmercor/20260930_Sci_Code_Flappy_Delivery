"""
Run the complete total-current spectral reconstruction using log-alpha continuation and return the percentage change in thermal conductivity relative to the fixed harmonic reconstruction.

The total-current reconstruction begins by subtracting the fixed harmonic imaginary-time correlation from the measured total-current correlation,

$$

\mathbf d=\mathbf C_{\mathrm{tot}}-\mathbf C_h.

$$

The Gaussian spectral kernel maps the positive additional spectrum into imaginary time. The regularized spectrum is then followed continuously in $\log\alpha$ until the prescribed chi-square discrepancy is reached.

The selected spectrum determines the additional zero-frequency contribution $\widetilde{\Lambda}(0)$, which is combined with the fixed harmonic zero-frequency spectrum to obtain the requested conductivity change,

$$

\Delta_\kappa=100\frac{\kappa_{\mathrm{tot}}-\kappa_h}{\kappa_h}.

$$

Returns
-------
float, the percentage change in thermal conductivity relative to the harmonic reconstruction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Run the complete total-current reconstruction in reduced units.

    Parameters
    ----------
    tau : np.ndarray
        Nonempty finite array of shape (n_obs,), with 0 <= tau[i] <= beta.
    beta : float
        Finite strictly positive inverse temperature.
    harmonic_correlation : np.ndarray
        Finite fixed harmonic correlation, shape (n_obs,).
    total_correlation : np.ndarray
        Finite measured total-current correlation, shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors, shape (n_obs,).
    centers : np.ndarray
        Nonempty finite nonnegative Gaussian centers, shape (n_basis,).
    widths : np.ndarray
        Finite strictly positive Gaussian widths, shape (n_basis,).
    default : np.ndarray
        Finite strictly positive default amplitudes, shape (n_basis,).
    harmonic_lambda_zero : float
        Finite strictly positive harmonic spectral density at zero frequency.
    target_chi2 : float
        Finite nonnegative discrepancy target.
    alpha_lower : float
        Finite strictly positive lower continuation bound.
    alpha_upper : float
        Finite upper continuation bound strictly greater than alpha_lower.
        The bounds must contain a spectrum meeting the discrepancy target.

    Returns
    -------
    delta_percent : float
        Percentage conductivity change relative to the fixed harmonic
        reconstruction. Compose the preceding public functions to obtain it.

    Raises
    ------
    ValueError
        If array shapes, finiteness or the stated domains are invalid, or
        if the continuation interval does not bracket the target discrepancy.
    RuntimeError
        If an upstream stationary solve or continuation refinement fails.
    """
    return delta_percent

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_total_current_reconstruction(
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

    kernel = _oracle_build_imaginary_time_kernel(
        tau,
        beta,
        centers,
        widths,
    )

    fit = _oracle_trace_discrepancy_continuation(
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
        _oracle_compute_zero_frequency_spectrum(
            amplitudes,
            centers,
            widths,
        )
    )

    conductivity = (
        _oracle_compute_conductivity_change(
            beta,
            harmonic_lambda_zero,
            additional_lambda_zero,
        )
    )

    return float(
        conductivity[2]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return comparisons with independent candidate/reference inputs."""
    return [
        {
            "setup": """import numpy as np
tau = np.array([0.075, 0.165, 0.285, 0.420, 0.570, 0.735], dtype=float)
beta = 1.5
harmonic_correlation = np.array([
    1.520000000000,
    1.470000000000,
    1.420000000000,
    1.380000000000,
    1.355000000000,
    1.345000000000,
], dtype=float)
total_correlation = np.array([
    1.589244048566,
    1.518820260964,
    1.454425796087,
    1.403371336138,
    1.373713928683,
    1.360610293767,
], dtype=float)
sigma = np.array([0.0014, 0.0012, 0.0010, 0.0009, 0.0008, 0.0008], dtype=float)
centers = np.array([0.35, 0.80, 1.40, 2.20, 3.20, 4.50, 6.00], dtype=float)
widths = np.array([0.18, 0.25, 0.32, 0.42, 0.55, 0.72, 0.95], dtype=float)
default = np.array([0.003, 0.006, 0.012, 0.020, 0.030, 0.025, 0.015], dtype=float)
harmonic_lambda_zero = 6.485568295814
target_chi2 = 6.0
alpha_lower = 1100.0
alpha_upper = 1200.0
""",
            "call": 'run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
            "gold_call": '_oracle_run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
        },
        {
            "setup": """import numpy as np
tau = np.array([0.10, 0.30, 0.60, 0.90], dtype=float)
beta = 1.2
harmonic_correlation = np.array([1.20, 1.10, 1.00, 0.95], dtype=float)
total_correlation = np.array([
    1.232680446815444,
    1.123387771997975,
    1.022332169574857,
    0.974487771997975,
], dtype=float)
sigma = np.array([0.0020, 0.0015, 0.0012, 0.0012], dtype=float)
centers = np.array([0.20, 1.00, 2.50], dtype=float)
widths = np.array([0.15, 0.30, 0.60], dtype=float)
default = np.array([0.01, 0.02, 0.03], dtype=float)
harmonic_lambda_zero = 4.2
target_chi2 = 3.0
alpha_lower = 1000.0
alpha_upper = 1150.0
""",
            "call": 'run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
            "gold_call": '_oracle_run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
        },
        {
            "setup": """import numpy as np
tau = np.array([0.0, 0.5, 1.0], dtype=float)
beta = 1.0
harmonic_correlation = np.array([0.80, 0.75, 0.80], dtype=float)
total_correlation = np.array([
    0.816731521285204,
    0.763981092126506,
    0.816731521285204,
], dtype=float)
sigma = np.array([0.003, 0.002, 0.003], dtype=float)
centers = np.array([0.0, 0.7], dtype=float)
widths = np.array([0.20, 0.25], dtype=float)
default = np.array([0.02, 0.03], dtype=float)
harmonic_lambda_zero = 2.5
target_chi2 = 1.0
alpha_lower = 600.0
alpha_upper = 700.0
""",
            "call": 'run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
            "gold_call": '_oracle_run_total_current_reconstruction(tau.copy(), beta, harmonic_correlation.copy(), total_correlation.copy(), sigma.copy(), centers.copy(), widths.copy(), default.copy(), harmonic_lambda_zero, target_chi2, alpha_lower, alpha_upper)',
        },
    ]
