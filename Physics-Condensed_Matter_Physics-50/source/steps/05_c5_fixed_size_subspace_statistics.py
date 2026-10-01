"""
Enumerate the fixed-size singular-mode subspaces and compute their source-defined joint, marginal, and sequential conditional selection probabilities.

A size-(d) subset (S) of the (D) ordered modes has weight proportional to \(\prod_{k\in S}[\max(s_k,10^{-12}s_0)]\) for the benchmark exponent (c=1).  Subsets are unordered and sampled without replacement.  The marginal inclusion probability, not a conditional or joint subset probability, supplies the reciprocal factor in the weighted projector; the sequential scan conditionals follow the cited dynamic-programming rule.

Returns
-------
Return lexicographic int64 subsets `(6,2)`, float64 joint probabilities `(6,)`, float64 marginal inclusion probabilities `(4,)`, and float64 sequential conditionals `(4,2)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_subspace_statistics(
    singular_values: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return subsets, joint probabilities, marginals, and scan conditionals.

    For the frozen ``D=4`` and ``retained_dimension=2`` contract, the shapes are
    ``(6,2)``, ``(6,)``, ``(4,)``, and ``(4,2)``.  Subsets are lexicographically
    ordered.  Conditional entry ``[k,m]`` is the probability of accepting mode
    ``k`` after ``m`` earlier acceptances; unreachable zero-over-zero states are
    represented by zero.  Raise ``ValueError`` unless the spectrum is a finite,
    nonnegative, nonincreasing four-vector and the retained dimension is two.
    """
    return subsets, joint_probabilities, inclusion_probabilities, conditional_probabilities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_compute_subspace_statistics(
    singular_values: "np.ndarray",
    retained_dimension: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    s = np.asarray(singular_values, dtype=np.float64)
    if s.shape != (4,) or not np.all(np.isfinite(s)):
        raise ValueError("singular_values must be a finite vector of length four")
    if retained_dimension != 2:
        raise ValueError("the retained dimension is fixed at two")
    if s[0] <= 0.0 or np.any(s < 0.0) or np.any(s[:-1] < s[1:]):
        raise ValueError("singular values must be nonnegative and nonincreasing")
    weights = np.maximum(s, 1.0e-12 * s[0])
    subsets = np.asarray(list(itertools.combinations(range(4), 2)), dtype=np.int64)
    products = np.prod(weights[subsets], axis=1)
    partition = float(np.sum(products))
    if not np.isfinite(partition) or partition <= 0.0:
        raise ValueError("subset weights must have positive finite normalization")
    joint = products / partition
    marginal = np.zeros(4, dtype=np.float64)
    for probability, subset in zip(joint, subsets):
        marginal[subset] += probability

    suffix = np.zeros((5, 3), dtype=np.float64)
    suffix[4, 2] = 1.0
    for k in range(3, -1, -1):
        for m in range(2, -1, -1):
            skip = suffix[k + 1, m]
            take = weights[k] * suffix[k + 1, m + 1] if m < 2 else 0.0
            suffix[k, m] = skip + take
    conditional = np.zeros((4, 2), dtype=np.float64)
    for k in range(4):
        for m in range(2):
            take = weights[k] * suffix[k + 1, m + 1]
            skip = suffix[k + 1, m]
            denom = take + skip
            conditional[k, m] = take / denom if denom > 0.0 else 0.0
    return subsets, joint, marginal, conditional

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = "import numpy as np\ndef _pack(value):\n    parts=[]\n    for item in value:\n        arr=np.asarray(item); parts.append(np.asarray([arr.ndim,*arr.shape],dtype=np.complex128)); parts.append(arr.astype(np.complex128).ravel())\n    return np.concatenate(parts)\n"
    return [
        {"setup": pack + "singular_values=np.array([.91,.37,.12,.025]); retained_dimension=2", "call": "_pack(compute_subspace_statistics(singular_values,retained_dimension))", "gold_call": "_pack(_oracle_compute_subspace_statistics(singular_values,retained_dimension))", "tol": 1e-12},
        {"setup": pack + "singular_values=np.array([1.,1.,1.,1.]); retained_dimension=2", "call": "_pack(compute_subspace_statistics(singular_values,retained_dimension))", "gold_call": "_pack(_oracle_compute_subspace_statistics(singular_values,retained_dimension))", "tol": 1e-12},
        {"setup": pack + "singular_values=np.array([2.,.2,.002,0.]); retained_dimension=2", "call": "_pack(compute_subspace_statistics(singular_values,retained_dimension))", "gold_call": "_pack(_oracle_compute_subspace_statistics(singular_values,retained_dimension))", "tol": 1e-12},
        {"setup": "import numpy as np\nsingular_values=np.array([1.,.5,.2,.1]); retained_dimension=1\ndef candidate_result():\n    try:\n        compute_subspace_statistics(singular_values,retained_dimension)\n        return 0\n    except ValueError:\n        return 1\ndef reference_result():\n    try:\n        _oracle_compute_subspace_statistics(singular_values,retained_dimension)\n        return 0\n    except ValueError:\n        return 1", "call": "candidate_result()", "gold_call": "reference_result()"},
    ]
