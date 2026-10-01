"""
Run the complete stability-audited moving-grid calculation and return its overlap-preserving relative physical error.

At each equally spaced audit time, assemble the affine operator, verify the two directional donor rows, and form $P=\\operatorname{blockdiag}(H_LX_L^{-1},H_MX_M^{-1},H_RX_R^{-1})$. Evaluate both frozen stability diagnostics and reject the calculation if any spectral abscissa is nonnegative; a positive weighted instantaneous-growth rate is allowed.



After time advancement, evaluate the exact translated Gaussian on all three final grids. With $J_\\kappa=\\operatorname{diag}(D_\\kappa x_\\kappa)$, retain every overlap copy and accumulate



$$

E=\\sqrt{\\frac{\\sum_\\kappa e_\\kappa^T J_\\kappa H_\\kappa e_\\kappa}{\\sum_\\kappa u_{\\star,\\kappa}^T J_\\kappa H_\\kappa u_{\\star,\\kappa}}}.

$$



The final step must directly call every earlier oracle twin in its reference path and return one finite scalar.

Returns
-------
one finite float equal to the relative composite physical SBP error
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_moving_overset_error(
    counts: np.ndarray = None,
    sigmas: np.ndarray = None,
    c: float = 1.0,
    amplitude: float = 0.1,
    period: float = 1.0,
    penalties: np.ndarray = None,
    dt: float = 0.001,
    final_time: float = 1.0,
    beta: float = 80.0,
    x0: float = -0.4,
    spectral_samples: int = 41,
) -> float:
    r"""Return the relative composite physical SBP error $E$.

    Parameters
    ----------
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``. ``None`` selects ``[41, 49, 45]``.
    sigmas : np.ndarray
        Stretching parameters. ``None`` selects ``[0.25, -0.15, 0.20]``.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three penalties, each at least $1/2$. ``None`` selects three values of
        $0.75$.
    dt : float
        Positive RK4 step size dividing ``final_time`` exactly.
    final_time : float
        Positive terminal time.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.
    spectral_samples : int
        Number of equally spaced frozen-system audit times.

    Returns
    -------
    float
        One finite relative composite-grid error.

    Raises
    ------
    ValueError
        If ``spectral_samples`` is below two; ``c`` or ``period`` is invalid;
        the grid-speed magnitude is not below ``abs(c)``; the directional
        coupling rows are inconsistent; a frozen system is not strictly
        stable; or the exact-state norm is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_spectral_samples(spectral_samples):
    if isinstance(spectral_samples, (bool, np.bool_)) or not isinstance(
        spectral_samples, (int, np.integer)
    ):
        raise ValueError("spectral_samples must be an integer")
    if int(spectral_samples) < 2:
        raise ValueError("spectral_samples must be at least two")
    return int(spectral_samples)


def _oracle_compute_moving_overset_error(
    counts: np.ndarray = None,
    sigmas: np.ndarray = None,
    c: float = 1.0,
    amplitude: float = 0.1,
    period: float = 1.0,
    penalties: np.ndarray = None,
    dt: float = 0.001,
    final_time: float = 1.0,
    beta: float = 80.0,
    x0: float = -0.4,
    spectral_samples: int = 41,
) -> float:
    """Reference end-to-end stability audit, integration, and norm reduction."""
    spectral_samples = _validated_spectral_samples(spectral_samples)
    if counts is None:
        counts = np.array([41, 49, 45])
    if sigmas is None:
        sigmas = np.array([0.25, -0.15, 0.20])
    if penalties is None:
        penalties = np.array([0.75, 0.75, 0.75])
    counts = np.asarray(counts)
    sigmas = np.asarray(sigmas, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if not np.isfinite(c) or float(c) == 0.0:
        raise ValueError("c must be finite and nonzero")
    if not np.isfinite(amplitude) or not np.isfinite(period) or float(period) <= 0.0:
        raise ValueError("amplitude must be finite and period must be positive")
    if abs(2.0 * np.pi * float(amplitude) / float(period)) >= abs(float(c)):
        raise ValueError("grid-speed magnitude must remain below abs(c)")

    operators = [_oracle_build_sbp63_operator(int(count)) for count in counts]
    starts = np.r_[0, np.cumsum(counts)]
    direction = 1 if float(c) > 0.0 else -1
    maximum_abscissa = -np.inf
    maximum_instantaneous_growth = -np.inf
    for time in np.linspace(0.0, float(final_time), spectral_samples):
        state = _oracle_compute_moving_grid_state(
            time, counts, sigmas, amplitude, period
        )
        velocity = float(state[0])
        nodes = state[1:]
        metric = _oracle_compute_relative_metric_diagonals(c, velocity, counts, nodes)
        augmented = _oracle_assemble_weak_overset_system(
            time,
            counts,
            sigmas,
            c,
            amplitude,
            period,
            penalties,
            beta,
            x0,
        )
        system = augmented[0]
        if not np.all(np.isfinite(augmented[1:])):
            raise ValueError("the affine-system time derivatives are not finite")
        if direction > 0:
            first_weights = _oracle_compute_donor_interpolation(
                nodes[starts[0] : starts[1]], nodes[starts[1]], 0
            )[0]
            second_weights = _oracle_compute_donor_interpolation(
                nodes[starts[1] : starts[2]], nodes[starts[2]], 0
            )[0]
            first_row = starts[1]
            first_columns = slice(starts[0], starts[1])
            first_expected = (
                penalties[1] * metric[first_row] * first_weights / operators[1][0][0, 0]
            )
            second_row = starts[2]
            second_columns = slice(starts[1], starts[2])
            second_expected = (
                penalties[2]
                * metric[second_row]
                * second_weights
                / operators[2][0][0, 0]
            )
        else:
            first_weights = _oracle_compute_donor_interpolation(
                nodes[starts[1] : starts[2]], nodes[starts[1] - 1], 0
            )[0]
            second_weights = _oracle_compute_donor_interpolation(
                nodes[starts[2] : starts[3]], nodes[starts[2] - 1], 0
            )[0]
            first_row = starts[1] - 1
            first_columns = slice(starts[1], starts[2])
            first_expected = (
                penalties[0]
                * metric[first_row]
                * first_weights
                / operators[0][0][-1, -1]
            )
            second_row = starts[2] - 1
            second_columns = slice(starts[2], starts[3])
            second_expected = (
                penalties[1]
                * metric[second_row]
                * second_weights
                / operators[1][0][-1, -1]
            )
        if not np.allclose(
            system[first_row, first_columns],
            first_expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("first directional donor coupling is inconsistent")
        if not np.allclose(
            system[second_row, second_columns],
            second_expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("second directional donor coupling is inconsistent")
        energy_diagonal = np.empty(int(np.sum(counts)), dtype=float)
        for grid in range(3):
            section = slice(starts[grid], starts[grid + 1])
            energy_diagonal[section] = np.diag(operators[grid][0]) / metric[section]
        diagnostics = _oracle_compute_weighted_stability_diagnostics(
            system[:, :-1], np.diag(energy_diagonal)
        )
        maximum_abscissa = max(maximum_abscissa, float(diagnostics[0]))
        maximum_instantaneous_growth = max(
            maximum_instantaneous_growth, float(diagnostics[1])
        )
    if not np.isfinite(maximum_abscissa) or maximum_abscissa >= 0.0:
        raise ValueError("a frozen system is not strictly stable")
    if not np.isfinite(maximum_instantaneous_growth):
        raise ValueError("the weighted instantaneous-growth audit is not finite")

    numerical = _oracle_advance_weak_overset_rk4(
        counts,
        sigmas,
        c,
        amplitude,
        period,
        penalties,
        dt,
        final_time,
        beta,
        x0,
    )
    final_state = _oracle_compute_moving_grid_state(
        final_time, counts, sigmas, amplitude, period
    )
    final_nodes = final_state[1:]
    exact = np.exp(
        -float(beta) * (final_nodes - float(x0) - float(c) * float(final_time)) ** 2
    )
    error = numerical - exact

    numerator = 0.0
    denominator = 0.0
    for grid in range(3):
        section = slice(starts[grid], starts[grid + 1])
        h_matrix, _, derivative = operators[grid][:3]
        jacobian = derivative @ final_nodes[section]
        weighted_norm = jacobian[:, None] * h_matrix
        numerator += float(error[section] @ weighted_norm @ error[section])
        denominator += float(exact[section] @ weighted_norm @ exact[section])
    if denominator <= 0.0:
        raise ValueError("exact-state norm must be positive")
    result = float(np.sqrt(numerator / denominator))
    if not np.isfinite(result):
        raise ValueError("relative error is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return positive, negative, static, and invalid whole-pipeline cases."""
    return [
        {
            "setup": """import numpy as np
counts = np.array([41, 49, 45])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.1, 1.0
penalties = np.array([0.75, 0.75, 0.75])
dt, final_time = 0.001, 1.0
""",
            "call": "compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
            "gold_call": "_oracle_compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time)",
        },
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.0, 0.0, 0.0])
c, amplitude, period = 1.0, 0.0, 1.0
penalties = np.array([0.6, 0.7, 0.8])
dt, final_time = 0.002, 0.2
spectral_samples = 9
""",
            "call": "compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, spectral_samples=spectral_samples)",
            "gold_call": "_oracle_compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, spectral_samples=spectral_samples)",
        },
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.4, -0.3, 0.15])
c, amplitude, period = 1.0, 0.05, 0.8
penalties = np.array([0.65, 0.7, 0.8])
dt, final_time = 0.002, 0.2
spectral_samples = 9
""",
            "call": "compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, spectral_samples=spectral_samples)",
            "gold_call": "_oracle_compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, spectral_samples=spectral_samples)",
        },
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.4, -0.3, 0.15])
c, amplitude, period = -1.0, 0.05, 0.8
penalties = np.array([0.65, 0.7, 0.8])
dt, final_time = 0.002, 0.2
beta, x0 = 70.0, 0.35
spectral_samples = 9
""",
            "call": "compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, beta, x0, spectral_samples)",
            "gold_call": "_oracle_compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time, beta, x0, spectral_samples)",
        },
        {
            "setup": """import numpy as np
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.2, 1.0
penalties = np.array([0.75, 0.75, 0.75])
dt, final_time = 0.002, 0.2
def run_model():
    try:
        compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_moving_overset_error(counts, sigmas, c, amplitude, period, penalties, dt, final_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
