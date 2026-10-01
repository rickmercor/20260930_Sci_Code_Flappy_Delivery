"""
Chain the sub-problem functions 01-08 end to end and return the idempotency measure of the iterate reached once a given multiplication budget has been spent.

For a fixed step location, spectrum size and multiplication budget, the final idempotency trace is a scalar function of the initial interior gap. On a bracket that has a stable component-polynomial selection sequence and opposite endpoint residuals, bisection gives a deterministic inverse of that map. One degree-eight application costs three non-scalar products, so each residual evaluation performs floor(budget/3) recursive applications before the trace is compared with the target.

Returns
-------
float: midpoint of the final bracket for the initial gap whose fixed-budget expansion reaches target_trace.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def critical_initial_gap(size: int = 120, mu: float = 0.35,
                         multiplications: int = 30,
                         target_trace: float = 0.1,
                         gap_lower: float = 2.5e-4,
                         gap_upper: float = 3.5e-4,
                         bisection_steps: int = 40,
                         drop_tolerance: float = 1.0e-6) -> float:
    """Invert the fixed-budget idempotency trace on a supplied gap bracket.

    Parameters
    ----------
    size : int
        Number of eigenvalues in the equidistant test spectrum.
    mu : float
        Step location strictly inside the unit interval.
    multiplications : int
        Non-scalar matrix-matrix multiplication budget, at least 3.
        Any remainder after division by three is left unused.
    target_trace : float
        Finite target value of the final idempotency trace.
    gap_lower, gap_upper : float
        Finite positive endpoints with gap_lower < gap_upper. Both gaps must
        define valid test spectra, and their final trace residuals must have
        opposite signs (an endpoint residual equal to zero is accepted).
    bisection_steps : int
        Positive number of bisection updates to perform.
    drop_tolerance : float
        Finite nonnegative threshold passed to every component-polynomial
        application. Zero disables intermediate-product filtering.

    Returns
    -------
    gap : float
        Midpoint of the final bisection bracket, as a native Python float.

    Raises
    ------
    ValueError
        If an integer parameter is invalid; if a floating parameter is not
        real and finite; if ``drop_tolerance`` is negative; if the gap
        endpoints are not positive and ordered; if either endpoint does not
        define a valid spectrum; or if the two endpoint residuals do not
        bracket zero.

    Notes
    -----
    This is the final orchestrating step. Each residual evaluation must call
    the public functions of sub-problems 01-08 and feed their returned values
    through the matrix recursion. Include every import needed by the function
    body. Do not substitute a separately coded scalar-spectrum recurrence.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_critical_initial_gap(size: int = 120, mu: float = 0.35,
                                 multiplications: int = 30,
                                 target_trace: float = 0.1,
                                 gap_lower: float = 2.5e-4,
                                 gap_upper: float = 3.5e-4,
                                 bisection_steps: int = 40,
                                 drop_tolerance: float = 1.0e-6) -> float:
    import numpy as np

    for name, value, minimum in (
        ("size", size, 2),
        ("multiplications", multiplications, 3),
        ("bisection_steps", bisection_steps, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    size = int(size)
    multiplications = int(multiplications)
    bisection_steps = int(bisection_steps)

    for name, value in (
        ("mu", mu),
        ("target_trace", target_trace),
        ("gap_lower", gap_lower),
        ("gap_upper", gap_upper),
        ("drop_tolerance", drop_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    mu = float(mu)
    target_trace = float(target_trace)
    left = float(gap_lower)
    right = float(gap_upper)
    drop_tolerance = float(drop_tolerance)
    if not (0.0 < left < right):
        raise ValueError("require 0 < gap_lower < gap_upper")
    if drop_tolerance < 0.0:
        raise ValueError("drop_tolerance must be nonnegative")

    def residual(gap):
        eigenvalues = _oracle_equidistant_step_spectrum(size, mu, gap)
        iterate = _oracle_symmetric_matrix_from_spectrum(eigenvalues)
        lam_lumo = mu - 0.5 * gap
        lam_homo = mu + 0.5 * gap

        for iteration in range(1, multiplications // 3 + 1):
            state = np.asarray(
                _oracle_select_family_member(lam_lumo, lam_homo, iteration),
                dtype=float,
            )
            index = int(round(float(state[0])))
            lam_lumo = float(state[1])
            lam_homo = float(state[2])

            coeffs = _oracle_sp8_family_coefficients(index)
            evaluation = _oracle_degree_eight_evaluation_coefficients(coeffs)
            scalars = _oracle_workspace_rearrangement_scalars(evaluation)
            iterate = _oracle_apply_degree_eight_polynomial(
                iterate, evaluation, scalars, drop_tolerance
            )

        return float(_oracle_idempotency_trace(iterate) - target_trace)

    f_left = residual(left)
    f_right = residual(right)
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if np.signbit(f_left) == np.signbit(f_right):
        raise ValueError("gap endpoints must bracket the target trace")

    for _ in range(bisection_steps):
        middle = 0.5 * (left + right)
        f_middle = residual(middle)
        if f_middle == 0.0:
            return float(middle)
        if np.signbit(f_middle) == np.signbit(f_left):
            left = middle
            f_left = f_middle
        else:
            right = middle
            f_right = f_middle

    return float(0.5 * (left + right))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: a six-application solve below the centre of the interval.
        {
            "setup": """import numpy as np
args = (24, 0.35, 18, 0.24241986825026815, 0.001, 0.02, 18)
""",
            "call": "critical_initial_gap(*args)",
            "gold_call": "_oracle_critical_initial_gap(*args)",
        },
        # Valid: a seven-application solve above the centre.
        {
            "setup": """import numpy as np
args = (30, 0.62, 21, 0.20597545209847343, 0.001, 0.02, 16)
""",
            "call": "critical_initial_gap(*args)",
            "gold_call": "_oracle_critical_initial_gap(*args)",
        },
        # Boundary: a budget remainder cannot buy another application.
        {
            "setup": """import numpy as np
base = (26, 0.48, 0.220187843294567, 0.002, 0.04, 14)
""",
            "call": ("float(critical_initial_gap(base[0], base[1], 20, *base[2:])"
                     " + 1000.0 * (critical_initial_gap(base[0], base[1], 20, *base[2:])"
                     " - critical_initial_gap(base[0], base[1], 18, *base[2:])))"),
            "gold_call": ("float(_oracle_critical_initial_gap(base[0], base[1], 20, *base[2:])"
                          " + 1000.0 * (_oracle_critical_initial_gap(base[0], base[1], 20, *base[2:])"
                          " - _oracle_critical_initial_gap(base[0], base[1], 18, *base[2:])))"),
        },
        # Boundary: one bisection update returns the midpoint of the retained
        # half-bracket, which tests the update direction.
        {
            "setup": """import numpy as np
args = (18, 0.72, 15, 0.21509670981623413, 0.005, 0.08, 1)
""",
            "call": "critical_initial_gap(*args)",
            "gold_call": "_oracle_critical_initial_gap(*args)",
        },
        # Edge: a short recursion on an asymmetric spectrum.
        {
            "setup": """import numpy as np
args = (20, 0.65, 15, 0.20, 0.003, 0.06, 12)
""",
            "call": "critical_initial_gap(*args)",
            "gold_call": "_oracle_critical_initial_gap(*args)",
        },
        # Invalid: endpoint traces do not bracket the target.
        {
            "setup": """import numpy as np
def run_model():
    try:
        critical_initial_gap(20, 0.4, 12, 100.0, 0.001, 0.01, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_critical_initial_gap(20, 0.4, 12, 100.0, 0.001, 0.01, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: boolean bisection count is not an integer contract value.
        {
            "setup": """import numpy as np
def run_model():
    try:
        critical_initial_gap(20, 0.4, 12, 0.2, 0.001, 0.02, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_critical_initial_gap(20, 0.4, 12, 0.2, 0.001, 0.02, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: filtering thresholds must be finite and nonnegative.
        {
            "setup": """import numpy as np
def run_model():
    try:
        critical_initial_gap(20, 0.4, 12, 0.2, 0.001, 0.02, 8, -1e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_critical_initial_gap(20, 0.4, 12, 0.2, 0.001, 0.02, 8, -1e-6)
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
