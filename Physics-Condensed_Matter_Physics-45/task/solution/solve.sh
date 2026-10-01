#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_spin_casimir(S: float, hbar: float) -> float:
    """Reference implementation."""
    return float(S * (S + 1.0) * hbar**2)

import numpy as np


def compute_bond_quantities(
    state: "np.ndarray",
    J: float,
) -> "np.ndarray":
    """Reference implementation."""
    state = np.asarray(state, dtype=float)

    N1 = state[0]
    Nz = state[2]
    Mz = state[3]

    C = 2.0 * N1 + Mz**2 - Nz**2
    H_sc = J * C

    return np.array([C, H_sc], dtype=float)

import numpy as np


def compute_dimer_rhs(
    t: float,
    state: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    state = np.asarray(state, dtype=float)
    N1, N2, Nz, Mz = state

    Q = compute_spin_casimir(S, hbar)
    C, H_sc = compute_bond_quantities(state, J)

    dN1 = (
        -2.0 * J * Nz * N2
        + 2.0 * eta * N1 * H_sc
        - eta * J * Q * (Q - Mz * Mz - Nz * Nz)
    )

    dN2 = (
        J * Nz * (2.0 * N1 + Q + Mz * Mz - Nz * Nz)
        + 2.0 * eta * N2 * H_sc
    )

    dNz = -2.0 * J * N2 + eta * Nz * (H_sc + J * Q)

    dMz = eta * Mz * (H_sc - J * Q)

    return np.array([dN1, dN2, dNz, dMz], dtype=float)

import numpy as np
from scipy.integrate import solve_ivp


def integrate_dimer(
    times: "np.ndarray",
    state0: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    times = np.asarray(times, dtype=float)
    state0 = np.asarray(state0, dtype=float)

    t_max = float(times[-1])

    if t_max == 0.0:
        return np.repeat(
            state0[None, :],
            times.size,
            axis=0,
        ).astype(float)

    unique_times, inverse = np.unique(times, return_inverse=True)

    solution = solve_ivp(
        lambda t, y: compute_dimer_rhs(
            t,
            y,
            J,
            eta,
            S,
            hbar,
        ),
        (0.0, t_max),
        state0,
        t_eval=unique_times,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y.T[inverse].astype(float)

import numpy as np
from scipy.integrate import solve_ivp


def _calibration_rhs_partials(
    state: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return the state Jacobian A and parameter-derivative matrix B."""
    N1, N2, Nz, Mz = np.asarray(state, dtype=float)

    Q = compute_spin_casimir(S, hbar)

    C = 2.0 * N1 + Mz * Mz - Nz * Nz
    K = 2.0 * N1 + Q + Mz * Mz - Nz * Nz

    A = J * np.array(
        [
            [
                2.0 * eta * (C + 2.0 * N1),
                -2.0 * Nz,
                -2.0 * N2
                - 4.0 * eta * N1 * Nz
                + 2.0 * eta * Q * Nz,
                4.0 * eta * N1 * Mz
                + 2.0 * eta * Q * Mz,
            ],
            [
                2.0 * Nz + 4.0 * eta * N2,
                2.0 * eta * C,
                K
                - 2.0 * Nz * Nz
                - 4.0 * eta * N2 * Nz,
                2.0 * Nz * Mz
                + 4.0 * eta * N2 * Mz,
            ],
            [
                2.0 * eta * Nz,
                -2.0,
                eta * (C + Q - 2.0 * Nz * Nz),
                2.0 * eta * Nz * Mz,
            ],
            [
                2.0 * eta * Mz,
                0.0,
                -2.0 * eta * Nz * Mz,
                eta * (C - Q + 2.0 * Mz * Mz),
            ],
        ],
        dtype=float,
    )

    dfdJ = np.array(
        [
            -2.0 * Nz * N2
            + 2.0 * eta * N1 * C
            - eta * Q * (Q - Mz * Mz - Nz * Nz),
            Nz * K + 2.0 * eta * N2 * C,
            -2.0 * N2 + eta * Nz * (C + Q),
            eta * Mz * (C - Q),
        ],
        dtype=float,
    )

    dfdeta = J * np.array(
        [
            2.0 * N1 * C
            - Q * (Q - Mz * Mz - Nz * Nz),
            2.0 * N2 * C,
            Nz * (C + Q),
            Mz * (C - Q),
        ],
        dtype=float,
    )

    B = np.column_stack((dfdJ, dfdeta))

    return A, B


def compute_calibration_residuals(
    params: "np.ndarray",
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    params = np.asarray(params, dtype=float)
    observation_times = np.asarray(
        observation_times,
        dtype=float,
    )
    observations = np.asarray(
        observations,
        dtype=float,
    )
    state0 = np.asarray(state0, dtype=float)

    J = float(params[0])
    eta = float(params[1])

    sensitivity0 = np.zeros((4, 2), dtype=float)

    augmented0 = np.concatenate(
        [
            state0,
            sensitivity0.ravel(),
        ]
    )

    def _augmented_rhs(t, augmented):
        state = augmented[:4]

        sensitivity = augmented[4:].reshape(4, 2)

        state_rhs = compute_dimer_rhs(
            t,
            state,
            J,
            eta,
            S,
            hbar,
        )

        A, B = _calibration_rhs_partials(
            state,
            J,
            eta,
            S,
            hbar,
        )

        sensitivity_rhs = A @ sensitivity + B

        return np.concatenate(
            [
                state_rhs,
                sensitivity_rhs.ravel(),
            ]
        )

    t_max = float(observation_times[-1])

    if t_max == 0.0:
        augmented = np.repeat(
            augmented0[None, :],
            observation_times.size,
            axis=0,
        )
    else:
        solution = solve_ivp(
            _augmented_rhs,
            (0.0, t_max),
            augmented0,
            t_eval=observation_times,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        augmented = solution.y.T

    trajectory = augmented[:, :4]

    sensitivities = augmented[:, 4:].reshape(
        observation_times.size,
        4,
        2,
    )

    predicted = np.empty(4, dtype=float)
    jacobian = np.empty((4, 2), dtype=float)

    predicted[0] = trajectory[0, 2]
    jacobian[0] = sensitivities[0, 2]

    predicted[1] = trajectory[1, 1]
    jacobian[1] = sensitivities[1, 1]

    predicted[2] = trajectory[2, 2]
    jacobian[2] = sensitivities[2, 2]

    last_state = trajectory[3]

    predicted[3] = compute_bond_quantities(
        last_state,
        J,
    )[0]

    dC_dstate = np.array(
        [
            2.0,
            0.0,
            -2.0 * last_state[2],
            2.0 * last_state[3],
        ],
        dtype=float,
    )

    jacobian[3] = (
        dC_dstate @ sensitivities[3]
    )

    residuals = predicted - observations

    chi2 = float(
        np.dot(
            residuals,
            residuals,
        )
    )

    return np.concatenate(
        [
            residuals,
            np.array([chi2], dtype=float),
            jacobian.ravel(),
        ]
    )

import numpy as np
from scipy.optimize import least_squares, shgo


def fit_dimer_parameters(
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
                compute_calibration_residuals(
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
        compute_calibration_residuals(
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

import numpy as np
from scipy.optimize import brentq


def find_first_correlation_crossing(
    J: float,
    eta: float,
    target: float,
    t_start: float,
    t_end: float,
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> float:
    """Reference implementation."""
    lower = np.nextafter(
        float(t_start),
        float(t_end),
    )

    n_grid = max(
        1001,
        int(
            np.ceil(
                (float(t_end) - lower) * 500.0
            )
        ) + 1,
    )

    grid = np.linspace(
        lower,
        float(t_end),
        n_grid,
    )

    trajectory = integrate_dimer(
        grid,
        state0,
        J,
        eta,
        S,
        hbar,
    )

    values = np.array(
        [
            compute_bond_quantities(
                row,
                J,
            )[0] - target
            for row in trajectory
        ],
        dtype=float,
    )

    bracket = None

    for i in range(grid.size - 1):
        if values[i] * values[i + 1] <= 0.0:
            bracket = (
                float(grid[i]),
                float(grid[i + 1]),
            )
            break

    if bracket is None:
        raise ValueError(
            "No correlation crossing found in the requested interval"
        )

    def _root_value(t):
        state = integrate_dimer(
            np.array([float(t)], dtype=float),
            state0,
            J,
            eta,
            S,
            hbar,
        )[0]

        return float(
            compute_bond_quantities(
                state,
                J,
            )[0] - target
        )

    return float(
        brentq(
            _root_value,
            bracket[0],
            bracket[1],
            xtol=1e-13,
            rtol=1e-13,
        )
    )

import numpy as np


def solve_dimer_calibration(
    S: float,
    hbar: float,
    state0: "np.ndarray",
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    J_bounds: tuple[float, float],
    eta_bounds: tuple[float, float],
    target: float,
    t_start: float,
    t_end: float,
) -> float:
    """Reference implementation."""
    fit = fit_dimer_parameters(
        J_bounds,
        eta_bounds,
        observation_times,
        observations,
        state0,
        S,
        hbar,
    )

    J_fit = float(fit[0])
    eta_fit = float(fit[1])

    crossing_time = find_first_correlation_crossing(
        J_fit,
        eta_fit,
        target,
        t_start,
        t_end,
        state0,
        S,
        hbar,
    )

    return float(crossing_time)
SCICODE_GOLD_EOF
