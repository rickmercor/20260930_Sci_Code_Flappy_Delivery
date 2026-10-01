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


def compute_modal_properties(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
) -> "np.ndarray":
    omega = np.asarray(omega, dtype=float)
    gamma_ph = np.asarray(gamma_ph, dtype=float)

    x = float(beta) * omega

    denominator = -np.expm1(-x)
    heat_capacity = x**2 * np.exp(-x) / denominator**2

    lifetime = 1.0 / (2.0 * gamma_ph)

    return np.vstack((heat_capacity, lifetime)).astype(float)

import numpy as np


def construct_spectral_components(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
) -> "np.ndarray":
    omega = np.asarray(omega, dtype=float)
    nu = np.asarray(nu, dtype=float)

    wn = omega[:, None]
    wm = omega[None, :]
    nu2 = nu**2
    beta = float(beta)

    singular_coeff = (
        nu2
        * (wn / np.expm1(beta * wn))
        * (wm / (-np.expm1(-beta * wm)))
        * ((wn + wm) ** 2 / (4.0 * wn * wm))
    )

    regular_coeff = (
        0.125
        * nu2
        * (wn / (-np.expm1(-beta * wn)))
        * (wm / (-np.expm1(-beta * wm)))
        * ((wn - wm) ** 2 / (2.0 * wn * wm))
    )

    singular_center = wm - wn
    regular_center = wn + wm

    pair_width = np.full_like(
        singular_coeff,
        2.0 * float(gamma_tr),
        dtype=float,
    )

    return np.stack(
        (
            singular_coeff,
            singular_center,
            regular_coeff,
            regular_center,
            pair_width,
        ),
        axis=0,
    ).astype(float)

import numpy as np
from scipy.special import exp1


def _lorentz_exp_integral(
    decay: float,
    center: "np.ndarray",
    width: "np.ndarray",
) -> "np.ndarray":
    z_plus = center + 1j * width
    z_minus = center - 1j * width

    value = (
        np.exp(-decay * z_plus) * exp1(-decay * z_plus)
        - np.exp(-decay * z_minus) * exp1(-decay * z_minus)
    ) / (2j)

    return np.real(value)


def transform_to_imaginary_time(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    tau: "np.ndarray",
) -> "np.ndarray":
    tau = np.asarray(tau, dtype=float)

    components = construct_spectral_components(
        omega,
        nu,
        beta,
        gamma_tr,
    )

    singular_coeff = components[0]
    singular_center = components[1]
    regular_coeff = components[2]
    regular_center = components[3]
    pair_width = components[4]

    values = []

    for tau_i in tau:
        singular_integral = (
            _lorentz_exp_integral(
                float(tau_i),
                singular_center,
                pair_width,
            )
            + _lorentz_exp_integral(
                float(beta) - float(tau_i),
                singular_center,
                pair_width,
            )
        )

        regular_integral = (
            _lorentz_exp_integral(
                float(tau_i),
                regular_center,
                pair_width,
            )
            + _lorentz_exp_integral(
                float(beta) - float(tau_i),
                regular_center,
                pair_width,
            )
        )

        c_s = float(
            np.sum(singular_coeff * singular_integral) / np.pi
        )

        c_r = float(
            np.sum(regular_coeff * regular_integral) / np.pi
        )

        values.append((c_s, c_r))

    return np.asarray(values, dtype=float).T

import numpy as np
from scipy.optimize import minimize_scalar


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
    tau = np.asarray(tau, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    gamma_lo, gamma_hi = map(float, gamma_bounds)
    xi_lo, xi_hi = map(float, xi_bounds)

    def _objective_and_xi(gamma):
        correlation = transform_to_imaginary_time(
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

import numpy as np


def evaluate_zero_frequency_spectrum(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    xi: float,
) -> "np.ndarray":
    components = construct_spectral_components(
        omega,
        nu,
        beta,
        gamma_tr,
    )

    singular_coeff = components[0]
    singular_center = components[1]
    regular_coeff = components[2]
    regular_center = components[3]
    pair_width = components[4]

    lambda_s = float(
        np.sum(
            singular_coeff
            * pair_width
            / (
                singular_center**2
                + pair_width**2
            )
        )
    )

    lambda_r = float(
        np.sum(
            regular_coeff
            * pair_width
            / (
                regular_center**2
                + pair_width**2
            )
        )
    )

    lambda_total = (
        lambda_s
        + float(xi) * lambda_r
    )

    tau_tr = (
        1.0
        / (2.0 * float(gamma_tr))
    )

    return np.array(
        [
            tau_tr,
            lambda_s,
            lambda_r,
            lambda_total,
        ],
        dtype=float,
    )

def compute_reconstructed_conductivity(
    beta: float,
    lambda_zero: float,
) -> float:
    return float(
        float(beta) ** 2
        * float(lambda_zero)
    )

import numpy as np


def compute_pb_rta_conductivity(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
) -> float:
    properties = compute_modal_properties(
        omega,
        beta,
        gamma_ph,
    )

    heat_capacity = properties[0]
    lifetime = properties[1]

    velocity = np.asarray(
        velocity,
        dtype=float,
    )

    return float(
        np.sum(
            heat_capacity
            * velocity**2
            * lifetime
        )
    )

def run_transport_reconstruction_pipeline(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    tau: "np.ndarray",
    c_obs: "np.ndarray",
    sigma: "np.ndarray",
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
    gamma_bounds: tuple[float, float],
    xi_bounds: tuple[float, float],
) -> float:
    fit = fit_transport_reconstruction(
        omega,
        nu,
        beta,
        tau,
        c_obs,
        sigma,
        gamma_bounds,
        xi_bounds,
    )

    gamma_tr = float(fit[0])
    xi = float(fit[1])

    zero_frequency = evaluate_zero_frequency_spectrum(
        omega,
        nu,
        beta,
        gamma_tr,
        xi,
    )

    lambda_zero = float(
        zero_frequency[3]
    )

    kappa_rec = compute_reconstructed_conductivity(
        beta,
        lambda_zero,
    )

    kappa_pb = compute_pb_rta_conductivity(
        omega,
        beta,
        gamma_ph,
        velocity,
    )

    return float(
        kappa_rec / kappa_pb
    )
SCICODE_GOLD_EOF
