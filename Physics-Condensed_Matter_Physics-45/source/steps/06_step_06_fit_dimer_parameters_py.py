"""
Determine the bounded global calibration minimum and refine it using the propagated residual-sensitivity Jacobian.

The calibration objective is nonlinear and can develop multiple local basins because the measured observables are taken from an oscillatory dissipative spin trajectory. A single local optimization from an arbitrary starting point therefore does not establish the required global minimum.

First search the complete rectangular parameter domain for the smallest unweighted residual objective. Then refine the selected basin using the four residuals together with their propagated sensitivities with respect to $J$ and $\eta$.

The fitted parameters are

$$

(J_*,\eta_*)=\operatorname*{argmin}_{J,\eta}\chi^2(J,\eta),

$$

within the supplied parameter bounds. The returned objective is the unweighted sum of the four squared residuals at this fitted pair.

Returns
-------
np.ndarray of shape (3,), containing $[J_*,\eta_*,\chi^2_{\min}]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_dimer_parameters(
    J_bounds: tuple[float, float],
    eta_bounds: tuple[float, float],
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Find the bounded global calibration minimum for J and eta.

    Parameters
    ----------
    J_bounds : tuple[float, float]
        Lower and upper bounds for J.
    eta_bounds : tuple[float, float]
        Lower and upper bounds for eta.
    observation_times : np.ndarray
        Length-4 calibration times.
    observations : np.ndarray
        Length-4 measurements ordered as Nz, N2, Nz, and C.
    state0 : np.ndarray
        Length-4 initial state ordered as [N1, N2, Nz, Mz].
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    fit : np.ndarray
        Length-3 array containing [J_star, eta_star, chi2_min].
    """
    return fit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import least_squares, shgo


def _oracle_fit_dimer_parameters(
    J_bounds: tuple[float, float],
    eta_bounds: tuple[float, float],
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    bounds = [
        tuple(map(float, J_bounds)),
        tuple(map(float, eta_bounds)),
    ]

    cache_x = None
    cache_data = None

    def _calibration_data(x):
        nonlocal cache_x, cache_data

        x = np.asarray(x, dtype=float)

        if (
            cache_x is None
            or not np.array_equal(x, cache_x)
        ):
            cache_x = x.copy()

            cache_data = (
                _oracle_compute_calibration_residuals(
                    x,
                    observation_times,
                    observations,
                    state0,
                    S,
                    hbar,
                )
            )

        return cache_data

    def _objective(x):
        return float(
            _calibration_data(x)[4]
        )

    global_result = shgo(
        _objective,
        bounds,
        n=24,
        iters=2,
        sampling_method="simplicial",
    )

    lower = np.array(
        [
            bounds[0][0],
            bounds[1][0],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            bounds[0][1],
            bounds[1][1],
        ],
        dtype=float,
    )

    def _residual_vector(x):
        return _calibration_data(x)[:4]

    def _residual_jacobian(x):
        return _calibration_data(x)[5:].reshape(
            4,
            2,
        )

    refined = least_squares(
        _residual_vector,
        np.asarray(
            global_result.x,
            dtype=float,
        ),
        jac=_residual_jacobian,
        bounds=(lower, upper),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=500,
    )

    final_data = (
        _oracle_compute_calibration_residuals(
            np.asarray(
                refined.x,
                dtype=float,
            ),
            observation_times,
            observations,
            state0,
            S,
            hbar,
        )
    )

    return np.array(
        [
            refined.x[0],
            refined.x[1],
            final_data[4],
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
obs = np.array([0.96653326, 0.89872454, -1.20908983, -3.52539157], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
        {
            "setup": """import numpy as np
times = np.array([2.20, 3.40, 5.10, 6.80], dtype=float)
obs = np.array([-1.18357937, -0.33562813, 1.32489032, -3.76664008], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
        {
            "setup": """import numpy as np
times = np.array([1.80, 3.00, 4.70, 6.20], dtype=float)
obs = np.array([-0.75485973, -0.12947933, 0.30523999, -3.73078066], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_fit_dimer_parameters((0.4, 1.8), (0.02, 0.15), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
    ]
