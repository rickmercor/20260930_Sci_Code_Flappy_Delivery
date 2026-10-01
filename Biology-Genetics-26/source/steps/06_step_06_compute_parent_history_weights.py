"""
Expand seven lineage configurations into the 26 donor-subset history weights.

Expand seven lineage configurations into the 26 donor-subset history weights.



The seven configurations retain 3, 2, 2, 2, 1, 1, and 1 lineages at the introgression time. Each retained lineage independently enters the donor parent with probability gamma. Multiplying the corresponding Bernoulli subset weight by its configuration probability accounts for both coalescent and parent-choice uncertainty. The returned vector uses the parent-history array order declared in the task.



Returns

-------

numpy.ndarray of 26 probabilities that sum to one

Returns
-------
numpy.ndarray of 26 probabilities that sum to one
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_parent_history_weights(
    configuration_probabilities: np.ndarray,
    gamma: float,
) -> np.ndarray:
    '''Compute all three-descendant parent-history weights.

    Parameters
    ----------
    configuration_probabilities : np.ndarray
        Seven finite nonnegative probabilities summing to one.
    gamma : float
        Finite probability in [0, 1].

    Returns
    -------
    parent_history_weights : np.ndarray
        Twenty-six normalized weights in the task-declared parent-history array order.

    Raises
    ------
    ValueError
        If the configuration vector is not seven finite nonnegative values summing to one or gamma is not a finite value in [0, 1].
    '''
    return parent_history_weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_compute_parent_history_weights(
    configuration_probabilities: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    probabilities = np.asarray(configuration_probabilities, dtype=float)
    if probabilities.shape != (7,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("configuration_probabilities must contain seven finite values")
    if np.any(probabilities < 0.0) or not np.isclose(
        np.sum(probabilities),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("configuration probabilities must be nonnegative and sum to one")
    if not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)):
        raise ValueError("gamma must be finite and numeric")
    gamma_value = float(gamma)
    if gamma_value < 0.0 or gamma_value > 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    lineage_counts = (3, 2, 2, 2, 1, 1, 1)
    weights = []
    for configuration_probability, lineage_count in zip(
        probabilities,
        lineage_counts,
    ):
        for subset_mask in range(1 << lineage_count):
            donor_count = subset_mask.bit_count()
            weights.append(
                float(configuration_probability)
                * gamma_value**donor_count
                * (1.0 - gamma_value) ** (lineage_count - donor_count)
            )
    result = np.asarray(weights, dtype=float)
    if not np.isclose(np.sum(result), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("parent-history weights failed normalization")
    return result

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
expected = np.array([
    0.0031275771443173084, 0.009293864936471358,
    0.009293864936471358, 0.027617520358950577,
    0.009293864936471358, 0.027617520358950577,
    0.027617520358950577, 0.08206784109632626,
    0.02105808522008813, 0.06257591446203092,
    0.06257591446203092, 0.1859497209662742,
    0.008175588079574605, 0.024294464335068305,
    0.024294464335068305, 0.072193093828012,
    0.008175588079574605, 0.024294464335068305,
    0.024294464335068305, 0.072193093828012,
    0.03576428852213515, 0.10627666456690109,
    0.009058675014813559, 0.026918633244278307,
    0.009058675014813559, 0.026918633244278307,
])
def check(value):
    if not np.allclose(value, expected, rtol=0.0, atol=1e-12):
        raise AssertionError("unexpected prompt-instance history weights")
    return 1
''',
            "call": "check(compute_parent_history_weights(configuration_probabilities, 0.7482114295606223))",
            "gold_call": "check(_oracle_compute_parent_history_weights(configuration_probabilities, 0.7482114295606223))",
        },
        {
            "setup": '''configuration_probabilities = np.array([0.07, 0.11, 0.13, 0.17, 0.19, 0.23, 0.10])
expected = np.zeros(26)
expected[[7, 11, 15, 19, 21, 23, 25]] = configuration_probabilities
def check(value):
    if not np.array_equal(value, expected):
        raise AssertionError("unexpected donor-boundary history ordering")
    return 2
''',
            "call": "check(compute_parent_history_weights(configuration_probabilities, 1.0))",
            "gold_call": "check(_oracle_compute_parent_history_weights(configuration_probabilities, 1.0))",
        },
        {
            "setup": '''configuration_probabilities = np.array([0, 0, 0, 0, 0, 0, 1.0])
def error_code(function):
    try:
        function(configuration_probabilities, 1.1)
    except ValueError:
        return 3
    except Exception:
        return 4
    return 0
''',
            "call": "error_code(compute_parent_history_weights)",
            "gold_call": "error_code(_oracle_compute_parent_history_weights)",
        },
    ]
