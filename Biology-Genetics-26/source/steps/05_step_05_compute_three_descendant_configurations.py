"""
Compute the seven lineage-configuration probabilities for a rooted three-descendant internal recipient.

For recipient topology ((A, B), C), A and B first spend an interval alone, then all surviving lineages enter a second ancestral interval. Kingman coalescence allows no, one, or two events before introgression. Distinguishing the possible first coalescing pairs gives the seven source-defined configurations.

Returns
-------
numpy.ndarray of seven probabilities that sum to one
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_three_descendant_configurations(
    first_interval: float,
    second_interval: float,
) -> np.ndarray:
    '''Evaluate the seven three-descendant lineage configurations.

    Parameters
    ----------
    first_interval : float
        Finite nonnegative interval ancestral only to the sister pair.
    second_interval : float
        Finite nonnegative interval ancestral to all three descendants.

    Returns
    -------
    configuration_probabilities : np.ndarray
        Seven probabilities in source order.

    Raises
    ------
    ValueError
        If either interval is not finite and numeric or is negative.
    '''
    return configuration_probabilities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_compute_three_descendant_configurations(
    first_interval: float,
    second_interval: float,
) -> np.ndarray:
    """Reference implementation."""
    for name, value in (
        ("first_interval", first_interval),
        ("second_interval", second_interval),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite and numeric")
        if float(value) < 0.0:
            raise ValueError(f"{name} must be nonnegative")

    exp_first = math.exp(-float(first_interval))
    exp_second = math.exp(-float(second_interval))
    exp_three_second = math.exp(-3.0 * float(second_interval))
    one_specific_event = 0.5 * (exp_second - exp_three_second)
    two_events_given_first_pair = (
        1.0 / 3.0 - 0.5 * exp_second + exp_three_second / 6.0
    )
    probabilities = np.asarray(
        [
            exp_first * exp_three_second,
            (1.0 - exp_first) * exp_second + exp_first * one_specific_event,
            exp_first * one_specific_event,
            exp_first * one_specific_event,
            (1.0 - exp_first) * (1.0 - exp_second)
            + exp_first * two_events_given_first_pair,
            exp_first * two_events_given_first_pair,
            exp_first * two_events_given_first_pair,
        ],
        dtype=float,
    )
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    probabilities /= np.sum(probabilities)
    if not np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("configuration probabilities failed normalization")
    return probabilities

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''expected = np.array([
    0.19592957412690937,
    0.3321596351104242,
    0.12895761057772323,
    0.12895761057772323,
    0.14204095308903625,
    0.035977308259091866,
    0.035977308259091866,
])
def check(value):
    if not np.allclose(value, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("unexpected prompt-instance configuration probabilities")
    return 1
''',
            "call": "check(compute_three_descendant_configurations(0.37, 0.42))",
            "gold_call": "check(_oracle_compute_three_descendant_configurations(0.37, 0.42))",
        },
        {
            "setup": '''expected = np.array([
    0.12245642825298195,
    0.18706443776921378,
    0.18706443776921378,
    0.18706443776921378,
    0.10545008614645887,
    0.10545008614645887,
    0.10545008614645887,
])
def check(value):
    if not np.allclose(value, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("unexpected sister-interval boundary probabilities")
    return 2
''',
            "call": "check(compute_three_descendant_configurations(0.0, 0.7))",
            "gold_call": "check(_oracle_compute_three_descendant_configurations(0.0, 0.7))",
        },
        {
            "setup": '''def error_code(function):
    try:
        function(-0.1, 0.4)
    except ValueError:
        return 3
    except Exception:
        return 4
    return 0
''',
            "call": "error_code(compute_three_descendant_configurations)",
            "gold_call": "error_code(_oracle_compute_three_descendant_configurations)",
        },
    ]
