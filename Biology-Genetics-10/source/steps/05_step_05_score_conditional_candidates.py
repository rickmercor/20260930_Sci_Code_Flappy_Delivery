"""
Evaluate candidate haplotypes under a lead-conditioned factor mixture.

Given normalized quadrature weights w_g and partner ALT probabilities q_j(f_g),

the conditional probability of configuration x is the weighted product-Bernoulli

sum over factor nodes: sum_g w_g product_j q_j(f_g)^x_j

[1-q_j(f_g)]^(1-x_j). This is the fixed-node approximation to the paper's

lead-conditioned latent-factor integral.

Inputs

------

configurations : binary partner configurations

mixture_weights : normalized conditioned factor weights

partner_probabilities : node-specific partner ALT probabilities

Returns

-------

conditional_probabilities : probability of each candidate configuration

Returns
-------
np.ndarray of shape (m,), candidate conditional probabilities as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def score_conditional_candidates(
    configurations: np.ndarray,
    mixture_weights: np.ndarray,
    partner_probabilities: np.ndarray,
) -> np.ndarray:
    '''Evaluate candidate probabilities under a conditioned product-Bernoulli mixture.

    Parameters
    ----------
    configurations : np.ndarray
        Binary array with one partner configuration per row.
    mixture_weights : np.ndarray
        One-dimensional normalized quadrature weights.
    partner_probabilities : np.ndarray
        ALT probabilities with one row per quadrature node.

    Returns
    -------
    conditional_probabilities : np.ndarray
        Conditional probability of every configuration.

    Raises
    ------
    ValueError
        If `configurations` is not a nonempty two-dimensional binary array; if
        `mixture_weights` is not a nonempty finite one-dimensional vector, has
        a negative entry, or does not sum to one within absolute tolerance
        `1e-10`; or if `partner_probabilities` does not have shape
        `(len(mixture_weights), configurations.shape[1])`, is nonfinite, or has
        an entry outside `[0, 1]`.
    '''
    return conditional_probabilities  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_score_conditional_candidates(
    configurations: np.ndarray,
    mixture_weights: np.ndarray,
    partner_probabilities: np.ndarray,
) -> np.ndarray:
    """Reference implementation of fixed-mixture candidate scoring."""
    candidates = np.asarray(configurations)
    weights = np.asarray(mixture_weights, dtype=float)
    probabilities = np.asarray(partner_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("configurations must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("configurations must be binary")
    if weights.ndim != 1 or weights.size < 1 or not np.isfinite(weights).all():
        raise ValueError("mixture_weights must be a nonempty finite vector")
    if probabilities.shape != (weights.size, candidates.shape[1]):
        raise ValueError("partner_probabilities shape does not match weights and configurations")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("partner_probabilities must lie in [0, 1]")
    if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("mixture_weights must be nonnegative and sum to one")

    candidates = candidates.astype(np.uint8)
    conditional_probabilities = np.empty(candidates.shape[0], dtype=float)
    for index, candidate in enumerate(candidates):
        component = np.prod(
            np.where(candidate[None, :] == 1, probabilities, 1.0 - probabilities),
            axis=1,
        )
        conditional_probabilities[index] = float(np.sum(weights * component))
    return conditional_probabilities

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """configurations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
mixture_weights = np.array([0.4, 0.6])
partner_probabilities = np.array([[0.2, 0.8], [0.7, 0.3]])
""",
            "call": "score_conditional_candidates(configurations, mixture_weights, partner_probabilities)",
            "gold_call": "_oracle_score_conditional_candidates(configurations, mixture_weights, partner_probabilities)",
        },
        {
            "setup": """configurations = np.array([[0], [1]], dtype=np.uint8)
mixture_weights = np.array([1.0])
partner_probabilities = np.array([[0.25]])
""",
            "call": "score_conditional_candidates(configurations, mixture_weights, partner_probabilities)",
            "gold_call": "_oracle_score_conditional_candidates(configurations, mixture_weights, partner_probabilities)",
        },
        {
            "setup": """configurations = np.array([[0, 1]], dtype=np.uint8)
mixture_weights = np.array([0.5, 0.5])
partner_probabilities = np.array([[0.2], [0.8]])
def run_model():
    try:
        score_conditional_candidates(configurations, mixture_weights, partner_probabilities)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_score_conditional_candidates(configurations, mixture_weights, partner_probabilities)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
