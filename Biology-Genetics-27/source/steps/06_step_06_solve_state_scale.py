"""
Solve the scalar adjustment associated with the supplied per-state numerical components.

Each row of state_components contains [a_i, b_i, c_i] and defines



v_i(s) = a_i * s**2 + b_i * s + c_i.



Use midpoint binary search on [0, 1]. At a midpoint s, compare mean(v_i(s)) with initial_scalar. If the mean is larger, replace the upper bound with s; otherwise replace the lower bound with s. Return the first midpoint whose absolute difference from initial_scalar is at most tolerance. Raise ValueError if no midpoint meets the tolerance within max_iterations.



Let sd_1 be the sample standard deviation of v_i(1), using denominator n - 1, and let sd_s be the corresponding sample standard deviation at the solved midpoint.



Return



[solved midpoint,

 initial_scalar * midpoint**2,

 initial_standard_error * sd_s / sd_1].

Returns
-------
state_scale : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_state_scale(
    initial_scalar: float,
    initial_standard_error: float,
    state_components: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    """Return a solved scale and its adjusted numerical summaries.

    Parameters
    ----------
    initial_scalar : float
        Initial point scalar for the current trait state.
    initial_standard_error : float
        Initial standard error for the current trait state.
    state_components : np.ndarray
        Per-state quadratic, linear, and constant components.
    tolerance : float
        Positive absolute objective tolerance.
    max_iterations : int
        Positive maximum number of midpoint iterations.

    Returns
    -------
    np.ndarray
        Length-three vector containing the solved scale, adjusted
        scalar, and adjusted standard error.
    """
    return np.empty(3, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_state_scale(
    initial_scalar: float,
    initial_standard_error: float,
    state_components: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    if (
        not isinstance(
            initial_scalar,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            initial_scalar,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "initial_scalar must be a real number"
        )

    scalar = float(initial_scalar)

    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(
            "initial_scalar must be finite and non-negative"
        )

    if (
        not isinstance(
            initial_standard_error,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            initial_standard_error,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "initial_standard_error must be a real number"
        )

    standard_error = float(
        initial_standard_error
    )

    if (
        not np.isfinite(standard_error)
        or standard_error < 0.0
    ):
        raise ValueError(
            "initial_standard_error must be finite "
            "and non-negative"
        )

    components = np.asarray(
        state_components,
        dtype=float,
    )

    if (
        components.ndim != 2
        or components.shape[0] < 2
        or components.shape[1] != 3
    ):
        raise ValueError(
            "state_components must have shape "
            "(n_states, 3), with n_states >= 2"
        )

    if not np.all(np.isfinite(components)):
        raise ValueError(
            "state_components must contain only finite values"
        )

    if (
        not isinstance(
            tolerance,
            (int, float, np.integer, np.floating),
        )
        or isinstance(
            tolerance,
            (bool, np.bool_),
        )
    ):
        raise ValueError(
            "tolerance must be a real number"
        )

    absolute_tolerance = float(tolerance)

    if (
        not np.isfinite(absolute_tolerance)
        or absolute_tolerance <= 0.0
    ):
        raise ValueError(
            "tolerance must be finite and positive"
        )

    if (
        not isinstance(
            max_iterations,
            (int, np.integer),
        )
        or isinstance(
            max_iterations,
            (bool, np.bool_),
        )
        or int(max_iterations) < 1
    ):
        raise ValueError(
            "max_iterations must be a positive integer"
        )

    def state_values(
        scale: float,
    ) -> np.ndarray:
        return (
            components[:, 0] * scale * scale
            + components[:, 1] * scale
            + components[:, 2]
        )

    mean_at_zero = float(
        np.mean(state_values(0.0))
    )

    mean_at_one = float(
        np.mean(state_values(1.0))
    )

    if (
        scalar
        < min(
            mean_at_zero,
            mean_at_one,
        )
        - absolute_tolerance
        or scalar
        > max(
            mean_at_zero,
            mean_at_one,
        )
        + absolute_tolerance
    ):
        raise ValueError(
            "initial_scalar is not bracketed on [0, 1]"
        )

    lower = 0.0
    upper = 1.0
    solved_scale = None

    for _ in range(int(max_iterations)):
        midpoint = 0.5 * (
            lower + upper
        )

        midpoint_mean = float(
            np.mean(
                state_values(midpoint)
            )
        )

        if (
            abs(
                midpoint_mean
                - scalar
            )
            <= absolute_tolerance
        ):
            solved_scale = midpoint
            break

        if midpoint_mean > scalar:
            upper = midpoint
        else:
            lower = midpoint

    if solved_scale is None:
        raise ValueError(
            "scale search did not converge"
        )

    values_at_one = state_values(1.0)

    values_at_scale = state_values(
        solved_scale
    )

    standard_deviation_at_one = float(
        np.std(
            values_at_one,
            ddof=1,
        )
    )

    standard_deviation_at_scale = float(
        np.std(
            values_at_scale,
            ddof=1,
        )
    )

    if (
        not np.isfinite(
            standard_deviation_at_one
        )
        or standard_deviation_at_one <= 0.0
        or not np.isfinite(
            standard_deviation_at_scale
        )
    ):
        raise ValueError(
            "state standard deviations must be finite, "
            "with a positive value at scale 1"
        )

    adjusted_scalar = float(
        scalar
        * solved_scale
        * solved_scale
    )

    adjusted_standard_error = float(
        standard_error
        * standard_deviation_at_scale
        / standard_deviation_at_one
    )

    result = np.array(
        [
            solved_scale,
            adjusted_scalar,
            adjusted_standard_error,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "state-scale result must be finite"
        )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for solve_state_scale."""
    return [
        {
            "setup": """import numpy as np

initial_scalar = 0.18675
initial_standard_error = 0.12

state_components = np.array([
    [0.40,  0.10, 0.020],
    [0.50, -0.02, 0.010],
    [0.30,  0.05, 0.030],
    [0.45,  0.00, 0.015],
], dtype=float)

tolerance = 1e-10
max_iterations = 80
""",
            "call": (
                "solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
            "gold_call": (
                "_oracle_solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
        },
        {
            "setup": """import numpy as np

initial_scalar = 0.09666666666666668
initial_standard_error = 0.05

state_components = np.array([
    [0.25,  0.08, 0.01],
    [0.35, -0.04, 0.02],
    [0.20,  0.02, 0.03],
], dtype=float)

tolerance = 1e-12
max_iterations = 20
""",
            "call": (
                "solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
            "gold_call": (
                "_oracle_solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
        },
        {
            "setup": """import numpy as np

initial_scalar = 0.14052
initial_standard_error = 0.08

state_components = np.array([
    [0.15,  0.010, 0.005],
    [0.18, -0.005, 0.010],
    [0.12,  0.020, 0.004],
    [0.20, -0.010, 0.006],
    [0.16,  0.000, 0.008],
], dtype=float)

tolerance = 1e-10
max_iterations = 80
""",
            "call": (
                "solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
            "gold_call": (
                "_oracle_solve_state_scale("
                "initial_scalar, initial_standard_error, "
                "state_components, tolerance, max_iterations)"
            ),
        },
        {
            "setup": """import numpy as np

initial_scalar = 1.0
initial_standard_error = 0.05

state_components = np.array([
    [0.25,  0.08, 0.01],
    [0.35, -0.04, 0.02],
    [0.20,  0.02, 0.03],
], dtype=float)

tolerance = 1e-10
max_iterations = 80

def run_model():
    try:
        solve_state_scale(
            initial_scalar,
            initial_standard_error,
            state_components,
            tolerance,
            max_iterations,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_state_scale(
            initial_scalar,
            initial_standard_error,
            state_components,
            tolerance,
            max_iterations,
        )
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
