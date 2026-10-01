"""
Continue a real Taylor series and evaluate its rational curvature.

The denominator is normalized to have constant coefficient one.  Matching the input series through the requested total degree fixes the denominator and numerator uniquely, after which analytic polynomial derivatives give the second derivative without finite differencing.

Returns
-------
native float equal to the second derivative of the normalized Pade approximant at the evaluation point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pade_second_derivative(coefficients: "np.ndarray", numerator_degree: int, denominator_degree: int, evaluation: float) -> float:
    '''Return the second derivative of a normalized Pade approximant.

    Parameters
    ----------
    coefficients : np.ndarray
        One-dimensional finite real Taylor coefficient array.  At least
        numerator_degree + denominator_degree + 1 coefficients are required;
        later entries, if present, are ignored.
    numerator_degree, denominator_degree : int
        Nonnegative numerator and denominator degrees.
    evaluation : float
        Finite real point at which to evaluate the second derivative.

    Returns
    -------
    curvature : float
        Native Python float containing the analytic second derivative of the
        normalized rational approximant, with denominator constant term one.
        The denominator coefficients are obtained from the direct square
        coefficient-matching system; no least-squares fallback is used.

    Raises
    ------
    ValueError
        If degrees, coefficient shape, coefficient finiteness, or evaluation
        are invalid; if the denominator system is rank deficient; or if the
        evaluated denominator is zero within 64 machine epsilons of its
        absolute-term scale.
    '''
    return curvature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pade_second_derivative(coefficients: "np.ndarray", numerator_degree: int, denominator_degree: int, evaluation: float) -> float:
    """Reference implementation."""
    for value, name in ((numerator_degree, "numerator_degree"), (denominator_degree, "denominator_degree")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    coefficients_array = np.asarray(coefficients)
    required = numerator_degree + denominator_degree + 1
    if coefficients_array.ndim != 1 or coefficients_array.size < required:
        raise ValueError("coefficients must be one-dimensional and long enough")
    used = np.asarray(coefficients_array[:required], dtype=float)
    if not np.all(np.isfinite(used)) or not np.isfinite(evaluation):
        raise ValueError("coefficients and evaluation must be finite")

    if denominator_degree == 0:
        denominator = np.array([1.0], dtype=float)
    else:
        system = np.empty((denominator_degree, denominator_degree), dtype=float)
        right_hand_side = np.empty(denominator_degree, dtype=float)
        for row, order in enumerate(
            range(numerator_degree + 1, numerator_degree + denominator_degree + 1)
        ):
            system[row, :] = [
                used[order - q] if order - q >= 0 else 0.0
                for q in range(1, denominator_degree + 1)
            ]
            right_hand_side[row] = -used[order]
        if np.linalg.matrix_rank(system) < denominator_degree:
            raise ValueError("the Pade denominator system is rank deficient")
        denominator = np.concatenate((
            np.array([1.0]), np.linalg.solve(system, right_hand_side)
        ))

    numerator = np.empty(numerator_degree + 1, dtype=float)
    for order in range(numerator_degree + 1):
        numerator[order] = sum(
            denominator[q] * used[order - q]
            for q in range(min(order, denominator_degree) + 1)
        )

    point = float(evaluation)
    def _polynomial_jet(values):
        value = sum(values[j] * point ** j for j in range(values.size))
        first = sum(j * values[j] * point ** (j - 1) for j in range(1, values.size))
        second = sum(
            j * (j - 1) * values[j] * point ** (j - 2)
            for j in range(2, values.size)
        )
        return value, first, second

    a_value, a_first, a_second = _polynomial_jet(numerator)
    b_value, b_first, b_second = _polynomial_jet(denominator)
    denominator_scale = max(
        1.0,
        sum(abs(denominator[j] * point ** j) for j in range(denominator.size)),
    )
    if abs(b_value) <= 64.0 * np.finfo(float).eps * denominator_scale:
        raise ValueError("the Pade denominator vanishes at the evaluation point")
    curvature = (
        a_second / b_value
        - (a_value * b_second + 2.0 * a_first * b_first) / (b_value * b_value)
        + 2.0 * a_value * b_first * b_first / (b_value ** 3)
    )
    return float(curvature)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
coefficients = np.array([1.0,1.0,1.0])
numerator_degree, denominator_degree, evaluation = 1, 1, 0.2
""",
            "call": "pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "gold_call": "_oracle_pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
coefficients = np.array([0.7,-0.2,0.5,-0.3,0.11])
numerator_degree, denominator_degree, evaluation = 4, 0, -0.6
""",
            "call": "pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "gold_call": "_oracle_pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
coefficients = np.array([1.0,-0.4,0.23,-0.11,0.071,-0.035,0.019])
numerator_degree, denominator_degree, evaluation = 3, 3, -0.75
""",
            "call": "pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "gold_call": "_oracle_pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
coefficients = np.array([1.0,1.0,1.0])
numerator_degree, denominator_degree, evaluation = 0, 2, 0.2
""",
            "call": "pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "gold_call": "_oracle_pade_second_derivative(coefficients.copy(), numerator_degree, denominator_degree, evaluation)",
            "tol": 2e-12,
        },
        {
            "setup": """import numpy as np
coefficients = np.zeros(5)
numerator_degree, denominator_degree, evaluation = 2, 2, 0.5
def catch_value_error(fn):
    try:
        fn(coefficients, numerator_degree, denominator_degree, evaluation)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "catch_value_error(pade_second_derivative)",
            "gold_call": "catch_value_error(_oracle_pade_second_derivative)",
        },
    ]
