"""
Infer the mode-independent transport damping rate and regular-spectrum scale by minimizing the weighted mismatch between the reconstructed and supplied imaginary-time heat-current correlations over the specified parameter bounds.

The default two-parameter spectral reconstruction gives an imaginary-time model of the form



$$

C_{\mathrm{model}}^h(\tau;\Gamma^{tr},ξ)

=

C_s^h(\tau;\Gamma^{tr})

+

ξ C_r^h(\tau;\Gamma^{tr}).

$$



The fitted parameters minimize

$$

\chi^2(\Gamma^{tr},ξ)=\sum_i\left[\frac{C_s^h(\tau_i;\Gamma^{tr})+ξ C_r^h(\tau_i;\Gamma^{tr})-C_{xx}^h(\tau_i)}{\sigma_i}\right]^2.

$$

For a fixed $\Gamma^{tr}$, the model is linear in $ξ$, so the weighted least-squares optimum in $ξ$ can be determined directly and constrained to the allowed interval. The resulting one-dimensional objective in $\Gamma^{tr}$ is then searched across the full prescribed range.

Returns
-------
np.ndarray of length 3 containing fitted Gamma_tr, fitted xi, and the minimum weighted chi-square
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_transport_reconstruction(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    tau: "np.ndarray",
    c_obs: "np.ndarray",
    sigma: "np.ndarray",
    gamma_bounds: tuple[float, float],
    xi_bounds: tuple[float, float],
) -> "np.ndarray":
    """Fit the two-parameter transport spectral reconstruction.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    tau : np.ndarray
        Imaginary times satisfying 0 < tau_i < beta.
    c_obs : np.ndarray
        Supplied imaginary-time heat-current correlation values.
    sigma : np.ndarray
        Positive standard errors corresponding to c_obs.
    gamma_bounds : tuple[float, float]
        Lower and upper bounds for Gamma_tr.
    xi_bounds : tuple[float, float]
        Lower and upper bounds for xi.

    Returns
    -------
    fit : np.ndarray
        Length-3 array containing fitted Gamma_tr, fitted xi,
        and the minimum weighted chi-square.

    Notes
    -----
    The supported input domain contains a nonzero regular spectral
    contribution over the fitted imaginary-time points.
    """
    return fit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar


def _oracle_fit_transport_reconstruction(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    tau: "np.ndarray",
    c_obs: "np.ndarray",
    sigma: "np.ndarray",
    gamma_bounds: tuple[float, float],
    xi_bounds: tuple[float, float],
) -> "np.ndarray":
    tau = np.asarray(tau, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    gamma_lo, gamma_hi = map(float, gamma_bounds)
    xi_lo, xi_hi = map(float, xi_bounds)

    def _objective_and_xi(gamma):
        correlation = _oracle_transform_to_imaginary_time(
            omega,
            nu,
            beta,
            float(gamma),
            tau,
        )

        c_s = correlation[0]
        c_r = correlation[1]

        denominator = np.sum((c_r / sigma) ** 2)

        xi = np.sum(
            c_r * (c_obs - c_s) / sigma**2
        ) / denominator

        xi = float(np.clip(xi, xi_lo, xi_hi))

        model = c_s + xi * c_r

        chi2 = float(
            np.sum(
                ((model - c_obs) / sigma) ** 2
            )
        )

        return chi2, xi

    def _objective(gamma):
        return _objective_and_xi(float(gamma))[0]

    gamma_grid = np.linspace(
        gamma_lo,
        gamma_hi,
        241,
        dtype=float,
    )

    chi_grid = np.asarray(
        [_objective(gamma) for gamma in gamma_grid],
        dtype=float,
    )

    candidates = [
        (float(chi_grid[0]), float(gamma_grid[0])),
        (float(chi_grid[-1]), float(gamma_grid[-1])),
    ]

    for j in range(1, gamma_grid.size - 1):
        if (
            chi_grid[j] <= chi_grid[j - 1]
            and chi_grid[j] <= chi_grid[j + 1]
        ):
            refined = minimize_scalar(
                _objective,
                bounds=(
                    float(gamma_grid[j - 1]),
                    float(gamma_grid[j + 1]),
                ),
                method="bounded",
                options={
                    "xatol": 1e-13,
                    "maxiter": 200,
                },
            )

            candidates.append(
                (
                    float(refined.fun),
                    float(refined.x),
                )
            )

    _, gamma_fit = min(
        candidates,
        key=lambda item: item[0],
    )

    chi2_fit, xi_fit = _objective_and_xi(gamma_fit)

    return np.array(
        [
            gamma_fit,
            xi_fit,
            chi2_fit,
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
            "setup": """
import numpy as np

omega = np.array([0.75, 1.20, 1.85, 2.55], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.20, 0.55, 0.20, 0.10],
    [0.55, 1.00, 0.50, 0.18],
    [0.20, 0.50, 0.78, 0.40],
    [0.10, 0.18, 0.40, 0.58]
], dtype=float)
nu_gold = nu.copy()

beta = 1.5

tau = np.array(
    [0.060, 0.150, 0.270, 0.420, 0.585, 0.750],
    dtype=float,
)
tau_gold = tau.copy()

c_obs = np.array([
    1.514473717829,
    1.462768236466,
    1.416173007618,
    1.376243100066,
    1.353349579459,
    1.345265301976
], dtype=float)
c_obs_gold = c_obs.copy()

sigma = np.array(
    [0.0018, 0.0016, 0.0014, 0.0012, 0.0011, 0.0011],
    dtype=float,
)
sigma_gold = sigma.copy()

gamma_bounds = (0.03, 0.30)
xi_bounds = (0.5, 2.5)
""",
            "call": "np.round(fit_transport_reconstruction(omega, nu, beta, tau, c_obs, sigma, gamma_bounds, xi_bounds), 6)",
            "gold_call": "np.round(_oracle_fit_transport_reconstruction(omega_gold, nu_gold, beta, tau_gold, c_obs_gold, sigma_gold, gamma_bounds, xi_bounds), 6)",
        },
        {
            "setup": """
import numpy as np

omega = np.array([0.8, 1.4], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.0, 0.3],
    [0.3, 0.8]
], dtype=float)
nu_gold = nu.copy()

beta = 1.2

tau = np.array([0.12, 0.36, 0.60], dtype=float)
tau_gold = tau.copy()

c_obs = np.array([
    0.987747280022827,
    0.970093034693898,
    0.964928825882618
], dtype=float)
c_obs_gold = c_obs.copy()

sigma = np.array([0.002, 0.002, 0.002], dtype=float)
sigma_gold = sigma.copy()

gamma_bounds = (0.03, 0.25)
xi_bounds = (0.5, 2.5)
""",
            "call": "np.round(fit_transport_reconstruction(omega, nu, beta, tau, c_obs, sigma, gamma_bounds, xi_bounds), 6)",
            "gold_call": "np.round(_oracle_fit_transport_reconstruction(omega_gold, nu_gold, beta, tau_gold, c_obs_gold, sigma_gold, gamma_bounds, xi_bounds), 6)",
        },
        {
            "setup": """
import numpy as np

omega = np.array([1.00, 1.02, 1.80], dtype=float)
omega_gold = omega.copy()

nu = np.array([
    [1.0, 0.5, 0.2],
    [0.5, 0.9, 0.4],
    [0.2, 0.4, 0.7]
], dtype=float)
nu_gold = nu.copy()

beta = 2.0

tau = np.array([0.10, 0.50, 1.00], dtype=float)
tau_gold = tau.copy()

c_obs = np.array([
    0.502142710540689,
    0.463360303598931,
    0.450633355019335
], dtype=float)
c_obs_gold = c_obs.copy()

sigma = np.array([0.0015, 0.0012, 0.0010], dtype=float)
sigma_gold = sigma.copy()

gamma_bounds = (0.03, 0.15)
xi_bounds = (0.5, 2.5)
""",
            "call": "np.round(fit_transport_reconstruction(omega, nu, beta, tau, c_obs, sigma, gamma_bounds, xi_bounds), 6)",
            "gold_call": "np.round(_oracle_fit_transport_reconstruction(omega_gold, nu_gold, beta, tau_gold, c_obs_gold, sigma_gold, gamma_bounds, xi_bounds), 6)",
        },
    ]
