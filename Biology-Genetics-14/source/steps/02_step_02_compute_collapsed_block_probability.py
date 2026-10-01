"""
Evaluate the approximate collapsed posterior gate for one LD block.

The block indicator zeta_b first decides whether every sparse effect in block b

is zero or whether its SNP-level spike-and-slab model is active. Given the common

block-removed residual r_-b, integrating a Gaussian slab with variance

sigma2 * eta_beta gives v_j = [x_j^T x_j / sigma2 +

1 / (sigma2 * eta_beta)]^-1, mu_j = v_j x_j^T r_-b / sigma2, and

logBF_j = 0.5 log(v_j) - 0.5 log(sigma2 * eta_beta) + mu_j^2 / (2 v_j).

For this coarse block comparison, the within-block Gram matrix is diagonalized,

so Delta_b = sum_j log[(1 - pi_j) + pi_j exp(logBF_j)] is formed with the same

r_-b for every SNP. The posterior block probability is

sigmoid(logit(pi_block) + Delta_b). This diagonal approximation belongs only to

the block gate; an active block is subsequently handled by an LD-aware

sequential scan.

Inputs

------

X_rot : rotated genotype matrix with one column per SNP

block_indices : ordered SNP indices belonging to block b

residual_without_block : common residual r_-b used by every block SNP

sigma2, eta_beta : residual variance and slab-to-residual variance ratio

inclusion_probabilities : SNP prior probabilities pi_j

pi_block : prior probability that block b is active

Returns

-------

log_evidence_increment : approximate collapsed increment Delta_b

block_probability : posterior probability that zeta_b equals one

log_bayes_factors : per-SNP integrated log Bayes factors

Linkage-disequilibrium blocks provide a regional sparsity layer, allowing an entire correlated region to be inactive before individual variants are considered. This step evaluates the block-level evidence used for that regional decision.

Returns
-------
length 2 + len(block_indices): the evidence increment, the block-on probability, then the per-SNP log Bayes factors in block order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_collapsed_block_probability(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual_without_block: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    pi_block: float,
) -> np.ndarray:
    """Compute approximate collapsed evidence and the block-on probability.

    Parameters
    ----------
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    block_indices : np.ndarray
        Unique SNP indices in one LD block.
    residual_without_block : np.ndarray
        Residual after restoring the current block contribution.
    sigma2 : float
        Positive residual variance.
    eta_beta : float
        Positive slab-to-residual variance ratio.
    inclusion_probabilities : np.ndarray
        SNP prior probabilities with shape (p,), all strictly between zero and one.
    pi_block : float
        Block prior probability strictly between zero and one.

    Raises
    ------
    ValueError
        If dimensions, indices, variances, or probabilities are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 vector whose first two entries are the evidence increment and
        block-on probability, followed by the per-SNP log Bayes factors.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_collapsed_block_probability(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual_without_block: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    pi_block: float,
) -> np.ndarray:
    """Reference implementation."""
    X = np.asarray(X_rot, dtype=np.float64)
    indices = np.asarray(block_indices)
    residual = np.asarray(residual_without_block, dtype=np.float64)
    probabilities = np.asarray(inclusion_probabilities, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X_rot must be a nonempty finite matrix")
    if residual.shape != (X.shape[0],) or not np.all(np.isfinite(residual)):
        raise ValueError("residual_without_block must be a finite vector of length n")
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("block_indices must be a nonempty integer vector")
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= X.shape[1]) or np.unique(indices).size != indices.size:
        raise ValueError("block_indices must be unique valid column indices")
    if probabilities.shape != (X.shape[1],) or not np.all(np.isfinite(probabilities)):
        raise ValueError("inclusion_probabilities must have shape (p,)")
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("all SNP probabilities must lie strictly between zero and one")
    try:
        residual_variance = float(sigma2)
        slab_ratio = float(eta_beta)
        block_prior = float(pi_block)
    except (TypeError, ValueError) as exc:
        raise ValueError("variance parameters and pi_block must be scalar") from exc
    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")
    if not np.isfinite(slab_ratio) or slab_ratio <= 0.0:
        raise ValueError("eta_beta must be positive and finite")
    if not np.isfinite(block_prior) or not 0.0 < block_prior < 1.0:
        raise ValueError("pi_block must lie strictly between zero and one")

    log_bayes_factors = np.empty(indices.size, dtype=np.float64)
    evidence_terms = np.empty(indices.size, dtype=np.float64)
    for offset, j in enumerate(indices):
        x_j = X[:, j]
        xtx = float(x_j @ x_j)
        xty = float(x_j @ residual)
        variance = 1.0 / (
            xtx / residual_variance + 1.0 / (residual_variance * slab_ratio)
        )
        mean = variance * xty / residual_variance
        log_bayes_factor = (
            0.5 * np.log(variance)
            - 0.5 * np.log(residual_variance * slab_ratio)
            + 0.5 * mean * mean / variance
        )
        log_bayes_factors[offset] = log_bayes_factor
        evidence_terms[offset] = np.logaddexp(
            np.log1p(-probabilities[j]),
            np.log(probabilities[j]) + log_bayes_factor,
        )

    log_evidence_increment = float(np.sum(evidence_terms))
    block_log_odds = (
        np.log(block_prior) - np.log1p(-block_prior) + log_evidence_increment
    )
    block_probability = np.exp(-np.logaddexp(0.0, -block_log_odds))
    return np.concatenate(
        (
            np.array(
                [log_evidence_increment, float(block_probability)],
                dtype=np.float64,
            ),
            log_bayes_factors,
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([[1.3416407864998738] * 3, [1.0488088481701516, -1.0488088481701516, 1.0488088481701516], [1.02469507659596, 1.02469507659596, -1.02469507659596], [0.6123724356957945, -0.6123724356957945, -0.6123724356957945], [0.0, 0.0, 0.0]])
indices = np.array([0, 1, 2])
residual = np.array([1.3, 0.4, -0.9, 1.0, -0.5])
pi = np.array([0.17495631821290467, 0.1750862681640398, 0.16868159439781952])
""",
            "call": "compute_collapsed_block_probability(X, indices, residual, 0.55, 0.85, pi, 0.38)",
            "gold_call": "_oracle_compute_collapsed_block_probability(X, indices, residual, 0.55, 0.85, pi, 0.38)",
        },
        {
            "setup": """import numpy as np
X = np.zeros((2, 1))
indices = np.array([0])
residual = np.array([2.0, -1.0])
pi = np.array([0.4])
""",
            "call": "compute_collapsed_block_probability(X, indices, residual, 1.0, 2.0, pi, 0.3)",
            "gold_call": "_oracle_compute_collapsed_block_probability(X, indices, residual, 1.0, 2.0, pi, 0.3)",
        },
        {
            "setup": """import numpy as np
X = np.ones((2, 2))
indices = np.array([0, 0])
residual = np.ones(2)
pi = np.array([0.2, 0.2])
def run_model():
    try:
        compute_collapsed_block_probability(X, indices, residual, 1.0, 1.0, pi, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_collapsed_block_probability(X, indices, residual, 1.0, 1.0, pi, 0.5)
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
