"""
Masked-token training learns a categorical distribution over the four phased

genotype states at corrupted loci. Phase-aware negative log-likelihood averages

-log p(y) over the masked sites, so confusing ``0|1`` with ``1|0`` is penalized

even though both states have alternate-allele dosage one. This is the scalar loss

used to judge the frozen toy imputation pass.

Inputs

------

pooled: rows containing sample index, variant index, target state, and four probabilities

Returns

-------

masked_nll: native float mean negative log-likelihood

Returns
-------
float, the phase-aware masked mean negative log-likelihood as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def score_masked_likelihood(pooled: "np.ndarray") -> float:
    """Compute mean phase-aware negative log-likelihood at masked sites.

    Parameters
    ----------
    pooled : np.ndarray
        Array of shape ``(n_masked, 7)`` with nonnegative integral sample and
        variant indices, target state 1 through 4, and four class probabilities.

    Raises
    ------
    ValueError
        If pooled is not a nonempty finite array of shape ``(n_masked, 7)``, if
        index or target columns violate their integral ranges, if probabilities
        are outside ``[0, 1]`` or do not sum to one within ``1e-10``, or if a
        target-class probability is zero.

    Returns
    -------
    masked_nll : float
        Mean negative natural-log probability of the true phased state.
    """
    return masked_nll  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_score_masked_likelihood(pooled: "np.ndarray") -> float:
    """Reference implementation."""
    pooled = np.asarray(pooled, dtype=float)
    if pooled.ndim != 2 or pooled.shape[1] != 7 or pooled.shape[0] == 0:
        raise ValueError("pooled must have nonempty shape (n_masked, 7)")
    if not np.all(np.isfinite(pooled)):
        raise ValueError("pooled must be finite")
    if not np.all(pooled[:, :3] == np.floor(pooled[:, :3])):
        raise ValueError("indices and target states must be integral")
    if np.any(pooled[:, :2] < 0):
        raise ValueError("sample and variant indices must be nonnegative")
    targets = pooled[:, 2].astype(np.int64)
    if np.any(targets < 1) or np.any(targets > 4):
        raise ValueError("target states must be from 1 through 4")
    probabilities = pooled[:, 3:]
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("each probability row must sum to one")
    target_probabilities = probabilities[np.arange(len(targets)), targets - 1]
    if np.any(target_probabilities <= 0.0):
        raise ValueError("target-class probabilities must be positive")
    return float(-np.log(target_probabilities).mean())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
pooled = np.array(
    [[0, 1, 2, 0.1, 0.6, 0.2, 0.1], [1, 3, 4, 0.2, 0.1, 0.1, 0.6]],
    dtype=float,
)
""",
            "call": "score_masked_likelihood(pooled)",
            "gold_call": "_oracle_score_masked_likelihood(pooled)",
        },
        {
            "setup": """import numpy as np
pooled = np.array([[0, 0, 1, 1.0, 0.0, 0.0, 0.0]])
""",
            "call": "score_masked_likelihood(pooled)",
            "gold_call": "_oracle_score_masked_likelihood(pooled)",
        },
        {
            "setup": """import numpy as np
pooled = np.array(
    [[0, 1, 2, 0.1, 0.6, 0.2, 0.1], [1, 3, 4, 0.2, 0.1, 0.1, 0.6]],
    dtype=float,
)
pooled[0, 3:] = [0.2, 0.2, 0.2, 0.2]
def run_model():
    try:
        score_masked_likelihood(pooled)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_score_masked_likelihood(pooled)
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
