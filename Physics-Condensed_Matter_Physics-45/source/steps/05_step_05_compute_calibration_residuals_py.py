"""
Compute the four calibration residuals, their unweighted sum of squares, and the residual sensitivities with respect to $J$ and $\eta$ by propagating the tangent dynamics.

For a trial parameter pair $(J,\eta)$, the dissipative dimer trajectory determines the four calibration residuals associated with $N^z(t_1)$, $N^2(t_2)$, $N^z(t_3)$, and $C(t_4)$.

To retain parameter information during the nonlinear evolution, propagate the forward sensitivities of the state with respect to $J$ and $\eta$. If the state equation is

$$

\dot{\mathbf y}=\mathbf f(\mathbf y;J,\eta),

$$

define the two sensitivity columns $S_J$ and $S_\eta$. They satisfy

$$

\dot S_J=A S_J+b_J,

$$

and

$$

\dot S_\eta=A S_\eta+b_\eta,

$$

where $A$ is the state Jacobian of the dimer dynamics and $b_J$ and $b_\eta$ are its parameter derivatives. Because the supplied initial state is independent of the fitted parameters, both sensitivity columns start from zero.

The observable sensitivities are then used to construct the $4\times2$ Jacobian of the calibration residual vector. The unweighted objective remains

$$

\chi^2=r_1^2+r_2^2+r_3^2+r_4^2.

$$

Returns
-------
np.ndarray of shape (13,), containing the four residuals, their unweighted sum of squares, and the row-major $4\times2$ residual-sensitivity Jacobian.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_calibration_residuals(
    params: "np.ndarray",
    observation_times: "np.ndarray",
    observations: "np.ndarray",
    state0: "np.ndarray",
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Compute calibration residuals, objective, and parameter sensitivities.

    Parameters
    ----------
    params : np.ndarray
        Length-2 array containing [J, eta].
    observation_times : np.ndarray
        Length-4 array corresponding in order to measurements of
        Nz, N2, Nz, and C.
    observations : np.ndarray
        Length-4 array of measured values in the same observable order.
    state0 : np.ndarray
        Length-4 initial state ordered as [N1, N2, Nz, Mz].
        The initial state is independent of J and eta.
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    calibration_data : np.ndarray
        Length-13 array containing
        [r1, r2, r3, r4, chi2,
         dr1_dJ, dr1_deta,
         dr2_dJ, dr2_deta,
         dr3_dJ, dr3_deta,
         dr4_dJ, dr4_deta].
    """
    return calibration_data

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

    Q = _oracle_compute_spin_casimir(S, hbar)

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


def _oracle_compute_calibration_residuals(
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

        state_rhs = _oracle_compute_dimer_rhs(
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

    predicted[3] = _oracle_compute_bond_quantities(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
params = np.array([0.83, 0.06], dtype=float)
times = np.array([0.70, 1.35, 2.40, 2.80], dtype=float)
obs = np.array([0.96653326, 0.89872454, -1.20908983, -3.52539157], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
        {
            "setup": """import numpy as np
params = np.array([1.55, 0.04], dtype=float)
times = np.array([2.20, 3.40, 5.10, 6.80], dtype=float)
obs = np.array([-1.18357937, -0.33562813, 1.32489032, -3.76664008], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
        {
            "setup": """import numpy as np
params = np.array([1.72, 0.025], dtype=float)
times = np.array([1.80, 3.00, 4.70, 6.20], dtype=float)
obs = np.array([-0.75485973, -0.12947933, 0.30523999, -3.73078066], dtype=float)
state0 = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "np.round(compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
            "gold_call": "np.round(_oracle_compute_calibration_residuals(params.copy(), times.copy(), obs.copy(), state0.copy(), 1.5, 1.0), 8)",
        },
    ]
