"""
Compute the classical Laguerre nodes and weighted derivatives simultaneously.



Use the source's modified Glaser--Liu--Rokhlin construction rather than a

Jacobi eigensolve followed by global polynomial evaluation.  The distinction

is consequential beyond the floating-point boundary where otherwise valid

root routines or separately evaluated exponential factors cease to return a

complete finite state.

Returns
-------
tuple of finite float arrays with shapes (n + 1,) and (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_glr_laguerre_state(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the augmented GLR nodes and weighted root derivatives.

    ``n`` must be a non-boolean integer of at least one.  The returned nodes
    contain zero followed by the increasing positive roots of the degree-``n``
    classical Laguerre polynomial.  The second vector contains the derivative
    of its exponentially weighted Laguerre function at those roots, generated
    as part of the same predictor/corrector construction.  Invalid or
    nonfinite states raise ``ValueError``.

    Parameters
    ----------
    n : int
        Number of positive Laguerre nodes.

    Returns
    -------
    nodes : np.ndarray
        Float vector of shape ``(n + 1,)`` beginning with zero.
    weighted_derivatives : np.ndarray
        Finite nonzero float vector of shape ``(n,)`` in root order.
    """
    return nodes, weighted_derivatives  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_glr_laguerre_state(
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference modified GLR predictor/corrector construction."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least one")

    machine_epsilon = np.finfo(float).eps

    def _evaluate_weighted_laguerre(x: float) -> tuple[float, float]:
        exponential = np.exp(-0.5 * x)
        laguerre = (1.0 - x) * exponential
        difference = -x * exponential
        derivative = -0.5 * laguerre - exponential
        for degree in range(1, n):
            next_difference = (degree * difference - x * laguerre) / (degree + 1.0)
            next_laguerre = laguerre + next_difference
            next_derivative = derivative - 0.5 * (laguerre + next_laguerre)
            laguerre = next_laguerre
            difference = next_difference
            derivative = next_derivative
        return float(laguerre), float(derivative)

    def _phase_integrate(theta: float, target: float, x: float) -> float:
        step_size = (target - theta) / 10.0
        for _ in range(10):
            phase_term = n + 0.5 - 0.25 * x
            first = -step_size / (
                np.sqrt(phase_term / x)
                + 0.25 * (1.0 / x - 0.25 / phase_term) * np.sin(2.0 * theta)
            )
            theta += step_size
            x += first
            phase_term = n + 0.5 - 0.25 * x
            second = -step_size / (
                np.sqrt(phase_term / x)
                + 0.25 * (1.0 / x - 0.25 / phase_term) * np.sin(2.0 * theta)
            )
            x += 0.5 * (second - first)
        return float(x)

    def _refine_initial_root(seed: float) -> tuple[float, float]:
        value, derivative = _evaluate_weighted_laguerre(seed)
        phase = np.arctan(np.sqrt(seed / (n + 0.5 - 0.25 * seed)) * derivative / value)
        root = _phase_integrate(float(phase), -0.5 * np.pi, seed)
        step = np.inf
        for _ in range(200):
            if abs(step) <= machine_epsilon and abs(value) <= machine_epsilon:
                break
            value, derivative = _evaluate_weighted_laguerre(root)
            step = value / derivative
            root -= step
        _, derivative = _evaluate_weighted_laguerre(root)
        return float(root), float(derivative)

    initial_count = min(20, n)
    roots = np.zeros(n, dtype=float)
    derivatives = np.zeros(n, dtype=float)
    seed = 1.0 / (2.0 * n + 1.0)
    for index in range(initial_count):
        seed, derivatives[index] = _refine_initial_root(seed)
        roots[index] = seed
        seed *= 1.1

    root = roots[initial_count - 1]
    order = 60 if n < 30 else 30
    for current in range(initial_count - 1, n - 1):
        if current == n - 6:
            order = 60

        displacement = _phase_integrate(0.5 * np.pi, -0.5 * np.pi, root) - root
        inverse_scale = 1.0 / displacement
        scale2 = inverse_scale**2
        scale3 = inverse_scale**3
        scale4 = inverse_scale**4
        recurrence_term = root * (n + 0.5 - 0.25 * root)
        root2 = root**2

        coefficients = np.zeros(order + 1, dtype=float)
        derivative_coefficients = np.zeros(order + 1, dtype=float)
        coefficients[1] = derivatives[current] / inverse_scale
        coefficients[2] = -0.5 * coefficients[1] / (inverse_scale * root)
        coefficients[3] = (
            -coefficients[2] / (inverse_scale * root)
            + (-(1.0 + recurrence_term) * coefficients[1] / (6.0 * scale2)) / root2
        )
        derivative_coefficients[:3] = (
            coefficients[1],
            2.0 * coefficients[2] * inverse_scale,
            3.0 * coefficients[3] * inverse_scale,
        )
        for k in range(2, order - 1):
            coefficients[k + 2] = (
                -root
                * (2.0 * k + 1.0)
                * (k + 1.0)
                * coefficients[k + 1]
                / inverse_scale
                - (k * k + recurrence_term) * coefficients[k] / scale2
                - (n + 0.5 - 0.5 * root) * coefficients[k - 1] / scale3
                + 0.25 * coefficients[k - 2] / scale4
            ) / (root2 * (k + 2.0) * (k + 1.0))
            derivative_coefficients[k + 1] = (
                (k + 2.0) * coefficients[k + 2] * inverse_scale
            )

        coefficients = coefficients[::-1]
        derivative_coefficients = derivative_coefficients[::-1]
        powers = np.ones(order + 1, dtype=float)
        powers[-1] = inverse_scale
        newton_step = np.inf
        for _ in range(10):
            if abs(newton_step) <= machine_epsilon:
                break
            newton_step = np.dot(coefficients, powers) / np.dot(
                derivative_coefficients, powers
            )
            displacement -= newton_step
            scaled_displacement = inverse_scale * displacement
            powers = np.concatenate(
                ([inverse_scale], np.cumprod(np.full(order, scaled_displacement)))
            )[::-1]

        root += displacement
        roots[current + 1] = root
        derivatives[current + 1] = np.dot(derivative_coefficients, powers)

    if (
        not np.all(np.isfinite(roots))
        or np.any(roots <= 0.0)
        or np.any(np.diff(roots) <= 0.0)
        or not np.all(np.isfinite(derivatives))
        or np.any(derivatives == 0.0)
    ):
        raise ValueError("modified GLR construction failed")
    nodes = np.concatenate(([0.0], roots))
    return nodes, derivatives

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, minimum-degree, high-degree, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
n = 7
""",
            "call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float))))(compute_glr_laguerre_state(n))",
            "gold_call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float))))(_oracle_compute_glr_laguerre_state(n))",
        },
        {
            "setup": """import numpy as np
n = 1
""",
            "call": "(lambda value: float(np.dot(value[1], np.arange(1, value[1].size + 1, dtype=float))))(compute_glr_laguerre_state(n))",
            "gold_call": "(lambda value: float(np.dot(value[1], np.arange(1, value[1].size + 1, dtype=float))))(_oracle_compute_glr_laguerre_state(n))",
        },
        {
            "setup": """import numpy as np
n = 1024
""",
            "call": "(lambda value: float(np.dot(value[1], np.arange(1, value[1].size + 1, dtype=float))))(compute_glr_laguerre_state(n))",
            "gold_call": "(lambda value: float(np.dot(value[1], np.arange(1, value[1].size + 1, dtype=float))))(_oracle_compute_glr_laguerre_state(n))",
        },
        {
            "setup": """n = True
def run_model():
    try:
        compute_glr_laguerre_state(n)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_glr_laguerre_state(n)
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
