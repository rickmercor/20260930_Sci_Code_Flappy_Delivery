"""
Invert the 26-history attachment mixture for the per-lineage donor probability.

Invert the 26-history attachment mixture for the per-lineage donor probability.



Every conditional parent-history probability is weighted by a coalescent configuration and an independent donor subset. The resulting expected attachment frequency is a polynomial of degree at most three in gamma. A scientifically identified estimate requires exactly one real root in the probability interval. The conditional vector and the history-weight vector use the same parent-history array order declared in the task.



Returns

-------

float, the unique per-lineage donor probability in [0, 1]

Returns
-------
float, the unique per-lineage donor probability in [0, 1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def estimate_three_descendant_gamma(
    attachment_count: int,
    joint_availability: int,
    configuration_probabilities: np.ndarray,
    parent_attachment_probabilities: np.ndarray,
) -> float:
    '''Solve the three-descendant attachment mixture for gamma.

    Parameters
    ----------
    attachment_count : int
        Nonnegative attachment count.
    joint_availability : int
        Positive availability not smaller than the attachment count.
    configuration_probabilities : np.ndarray
        Seven finite probabilities summing to one.
    parent_attachment_probabilities : np.ndarray
        Twenty-six finite conditional probabilities in [0, 1], in the task-declared parent-history array order.

    Returns
    -------
    gamma_hat : float
        The unique model root in [0, 1].

    Raises
    ------
    ValueError
        If the counts are not valid nonnegative integers with positive joint availability, the configuration or conditional vectors violate their declared probability contracts, or the attachment equation does not have exactly one root in [0, 1].
    '''
    return gamma_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_estimate_three_descendant_gamma(
    attachment_count: int,
    joint_availability: int,
    configuration_probabilities: np.ndarray,
    parent_attachment_probabilities: np.ndarray,
) -> float:
    """Reference implementation."""
    if not isinstance(attachment_count, (int, np.integer)) or attachment_count < 0:
        raise ValueError("attachment_count must be a nonnegative integer")
    if not isinstance(joint_availability, (int, np.integer)) or joint_availability <= 0:
        raise ValueError("joint_availability must be a positive integer")
    if attachment_count > joint_availability:
        raise ValueError("attachment_count cannot exceed joint_availability")

    configurations = np.asarray(configuration_probabilities, dtype=float)
    conditionals = np.asarray(parent_attachment_probabilities, dtype=float)
    if configurations.shape != (7,) or not np.all(np.isfinite(configurations)):
        raise ValueError("configuration_probabilities must contain seven finite values")
    if np.any(configurations < 0.0) or not np.isclose(
        np.sum(configurations),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("configuration probabilities must be nonnegative and sum to one")
    if conditionals.shape != (26,) or not np.all(np.isfinite(conditionals)):
        raise ValueError("parent_attachment_probabilities must contain 26 finite values")
    if np.any((conditionals < 0.0) | (conditionals > 1.0)):
        raise ValueError("conditional probabilities must lie in [0, 1]")

    gamma_polynomial = np.polynomial.Polynomial([0.0, 1.0])
    non_donor_polynomial = np.polynomial.Polynomial([1.0, -1.0])
    expected_polynomial = np.polynomial.Polynomial([0.0])
    lineage_counts = (3, 2, 2, 2, 1, 1, 1)
    conditional_index = 0
    for configuration_probability, lineage_count in zip(
        configurations,
        lineage_counts,
    ):
        for subset_mask in range(1 << lineage_count):
            donor_count = subset_mask.bit_count()
            subset_polynomial = (
                gamma_polynomial**donor_count
                * non_donor_polynomial ** (lineage_count - donor_count)
            )
            expected_polynomial += (
                float(configuration_probability)
                * float(conditionals[conditional_index])
                * subset_polynomial
            )
            conditional_index += 1

    observed_frequency = float(attachment_count) / float(joint_availability)
    equation = expected_polynomial - np.polynomial.Polynomial([observed_frequency])
    coefficients = np.trim_zeros(equation.coef, trim="b")
    if coefficients.size <= 1:
        raise ValueError("the attachment frequency does not identify gamma")
    roots = np.polynomial.Polynomial(coefficients).roots()
    admissible = []
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10:
            candidate = float(np.real(root))
            if -1e-10 <= candidate <= 1.0 + 1e-10:
                candidate = min(1.0, max(0.0, candidate))
                if abs(float(expected_polynomial(candidate)) - observed_frequency) <= 1e-8:
                    if not any(abs(candidate - prior) <= 1e-10 for prior in admissible):
                        admissible.append(candidate)
    if len(admissible) != 1:
        raise ValueError("the inputs must yield exactly one root in [0, 1]")
    return float(admissible[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''configuration_probabilities = np.array([
    0.19592957412690937, 0.3321596351104242, 0.12895761057772323,
    0.12895761057772323, 0.14204095308903625,
    0.035977308259091866, 0.035977308259091866,
])
conditionals = np.array([
    0.08, 0.36, 0.43, 0.70, 0.31, 0.62, 0.68, 0.91,
    0.12, 0.58, 0.47, 0.88, 0.10, 0.51, 0.61, 0.89,
    0.11, 0.56, 0.49, 0.87, 0.15, 0.84, 0.13, 0.82, 0.14, 0.80,
])
expected = 0.7482114295606223
def check(value):
    result = float(value)
    if abs(result - expected) > 1e-8:
        raise AssertionError("unexpected prompt-instance gamma")
    return 1
''',
            "call": "check(estimate_three_descendant_gamma(50, 72, configuration_probabilities, conditionals))",
            "gold_call": "check(_oracle_estimate_three_descendant_gamma(50, 72, configuration_probabilities, conditionals))",
        },
        {
            "setup": '''configuration_probabilities = np.array([0, 0, 0, 0, 1.0, 0, 0])
conditionals = np.zeros(26)
conditionals[21] = 1.0
expected = 0.3
def check(value):
    result = float(value)
    if abs(result - expected) > 1e-12:
        raise AssertionError("unexpected linear-mixture gamma")
    return 2
''',
            "call": "check(estimate_three_descendant_gamma(3, 10, configuration_probabilities, conditionals))",
            "gold_call": "check(_oracle_estimate_three_descendant_gamma(3, 10, configuration_probabilities, conditionals))",
        },
        {
            "setup": '''configuration_probabilities = np.array([0, 1.0, 0, 0, 0, 0, 0])
conditionals = np.zeros(26)
conditionals[8] = 0.16
conditionals[11] = 0.16
def error_code(function):
    try:
        function(1, 10, configuration_probabilities, conditionals)
    except ValueError:
        return 3
    except Exception:
        return 4
    return 0
''',
            "call": "error_code(estimate_three_descendant_gamma)",
            "gold_call": "error_code(_oracle_estimate_three_descendant_gamma)",
        },
    ]
