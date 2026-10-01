"""
Rank the admissible haplotype matrices of a window by combining their read likelihood, their error-correction cost and their agreement with the phasings decoded on the graph.

Likelihood rewards a candidate that explains the fragments under the error model, error correction counts the allele flips a hard assignment of each fragment to its best haplotype would need, and agreement rewards consistency with the phasings the graphical model already decoded. The three quantities live on different scales, so each is rescaled inside the candidate set before they are combined.

Returns
-------
np.ndarray of shape (n_candidates,), float: the combined score of each candidate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def score_phasing_candidates(candidates: np.ndarray, positions: np.ndarray,
                             reads: np.ndarray, neighbor_pairs: np.ndarray,
                             neighbor_phasings: np.ndarray, error_rate: float,
                             weights: np.ndarray) -> np.ndarray:
    """Score the admissible haplotype matrices of one window.

    Every fragment with a called allele at one or more window positions
    contributes. For such a fragment let d_k be the number of those called
    alleles that differ from haplotype k of the candidate and let c be the
    number of them. The likelihood term sums the logarithm of the sum over k of
    the per-base error model raised to d_k times its complement raised to
    c - d_k. The error-correction term sums the smallest d_k. The agreement
    term counts the supplied neighbour vertices whose two SNPs both lie in the
    window and whose decoded phasing equals, as a multiset of haplotype rows,
    the candidate restricted to those two SNPs.

    Each of the three terms is rescaled to the unit interval across the
    candidate set, a term that takes a single value across the set being
    rescaled to zero throughout. The score adds the rescaled likelihood and
    agreement terms with their weights and subtracts the rescaled
    error-correction term with its weight.

    Parameters
    ----------
    candidates : np.ndarray
        Integer array of shape (n_candidates, K, P) with entries in {0, 1}.
    positions : np.ndarray
        One-dimensional integer array of the P distinct SNP indices of the
        window, in the column order of ``candidates``.
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0, 1 and -1 for reference, alternate and uncalled.
    neighbor_pairs : np.ndarray
        Integer array of shape (m, 2) holding the SNP pair of each neighbour
        vertex whose decoded phasing is supplied.
    neighbor_phasings : np.ndarray
        Integer array of shape (m, K, 2) holding the decoded phasing of each of
        those vertices, in the column order of ``neighbor_pairs``.
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.
    weights : np.ndarray
        One-dimensional array of three finite non-negative floats holding, in
        order, the likelihood weight, the error-correction weight and the
        agreement weight.

    Returns
    -------
    scores : np.ndarray
        Array of shape (n_candidates,) of floats, the score of each candidate
        in the order supplied.

    Raises
    ------
    ValueError
        If ``candidates`` is not a non-empty three-dimensional integer-valued
        array with entries in {0, 1}, if ``positions`` is not a one-dimensional
        array of distinct valid SNP indices whose length matches the candidate
        width, if ``reads`` is not a two-dimensional integer-valued array with
        entries in {-1, 0, 1}, if ``neighbor_pairs`` is not a two-column array
        of finite integer-valued valid SNP indices naming two distinct SNPs per
        row, if ``neighbor_phasings`` does not have shape (m, K, 2) with entries
        in {0, 1} and the same leading length as ``neighbor_pairs``, if
        ``error_rate`` is not a finite number strictly between zero and one, or
        if ``weights`` is not a one-dimensional array of three finite
        non-negative entries.
    """
    return scores  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_score_phasing_candidates(candidates: np.ndarray, positions: np.ndarray,
                                     reads: np.ndarray, neighbor_pairs: np.ndarray,
                                     neighbor_phasings: np.ndarray, error_rate: float,
                                     weights: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    matrices = np.asarray(candidates, dtype=float)
    if matrices.ndim != 3 or matrices.size < 1:
        raise ValueError("candidates must be a non-empty three-dimensional array")
    if not np.all(np.isin(matrices, (0.0, 1.0))):
        raise ValueError("candidates entries must be 0 or 1")
    matrices = matrices.astype(int)
    ploidy = int(matrices.shape[1])

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    window = np.asarray(positions, dtype=float)
    if window.ndim != 1 or window.size != matrices.shape[2]:
        raise ValueError("positions must be one-dimensional and match the candidate width")
    if not np.allclose(window, np.round(window), rtol=0.0, atol=1e-12):
        raise ValueError("positions entries must be integer valued")
    window = np.round(window).astype(int)
    if np.unique(window).size != window.size:
        raise ValueError("positions entries must be distinct")
    if np.any(window < 0) or np.any(window >= fragments.shape[1]):
        raise ValueError("positions entries must be valid SNP indices")

    pairs = np.asarray(neighbor_pairs, dtype=float)
    if pairs.size == 0:
        pairs = pairs.reshape(0, 2)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("neighbor_pairs must be a two-dimensional array with two columns")
    decoded = np.asarray(neighbor_phasings, dtype=float)
    if decoded.size == 0:
        decoded = decoded.reshape(0, ploidy, 2)
    if decoded.ndim != 3 or decoded.shape[1] != ploidy or decoded.shape[2] != 2:
        raise ValueError("neighbor_phasings must have shape (m, K, 2) for the candidate ploidy")
    if pairs.shape[0] != decoded.shape[0]:
        raise ValueError("neighbor_pairs and neighbor_phasings must have the same length")
    if pairs.size and not (np.all(np.isfinite(pairs))
                           and np.allclose(pairs, np.round(pairs), rtol=0.0, atol=1e-12)):
        raise ValueError("neighbor_pairs entries must be finite and integer valued")
    if decoded.size and not np.all(np.isin(decoded, (0.0, 1.0))):
        raise ValueError("neighbor_phasings entries must be 0 or 1")
    pairs = np.round(pairs).astype(int)
    decoded = decoded.astype(int)
    if pairs.size and (np.any(pairs < 0) or np.any(pairs >= fragments.shape[1])):
        raise ValueError("neighbor_pairs entries must be valid SNP indices")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("each neighbour vertex must name two distinct SNPs")

    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    epsilon = float(error_rate)

    coefficients = np.asarray(weights, dtype=float).ravel()
    if coefficients.size != 3 or not np.all(np.isfinite(coefficients)) \
            or np.any(coefficients < 0.0):
        raise ValueError("weights must hold three finite non-negative entries")

    def _canonical(matrix):
        return matrix[np.lexsort(matrix[:, ::-1].T)]

    observed = fragments[:, window]
    observed = observed[np.any(observed >= 0, axis=1)]
    order = window.tolist()

    n_candidates = int(matrices.shape[0])
    likelihood = np.zeros(n_candidates, dtype=float)
    correction = np.zeros(n_candidates, dtype=float)
    agreement = np.zeros(n_candidates, dtype=float)
    for i, candidate in enumerate(matrices):
        for fragment in observed:
            called = fragment >= 0
            distance = (candidate[:, called] != fragment[None, called]).sum(axis=1)
            covered = int(called.sum())
            likelihood[i] += float(np.log(np.sum(
                epsilon ** distance * (1.0 - epsilon) ** (covered - distance))))
            correction[i] += float(distance.min())
        matched = 0
        for j in range(pairs.shape[0]):
            first, second = int(pairs[j, 0]), int(pairs[j, 1])
            if first in order and second in order:
                projection = candidate[:, [order.index(first), order.index(second)]]
                if np.array_equal(_canonical(projection), _canonical(decoded[j])):
                    matched += 1
        agreement[i] = float(matched)

    def _rescale(values):
        low, high = float(values.min()), float(values.max())
        if high - low <= 0.0:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    return (coefficients[0] * _rescale(likelihood)
            - coefficients[1] * _rescale(correction)
            + coefficients[2] * _rescale(agreement))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: six rival placements of one free column with the published
        #     weights and one decoded neighbour (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
import itertools
base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
cols = [np.array(p, dtype=int) for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
candidates = np.array([np.concatenate([base, c[:, None]], axis=1) for c in cols], dtype=int)
positions = np.array([0, 1, 2])
reads = np.array([[0, 0, 1, -1],
                  [0, 1, 0, -1],
                  [1, 0, 1, -1],
                  [1, 1, 0, -1],
                  [-1, 0, 1, 0],
                  [-1, 1, 0, 1]], dtype=int)
neighbor_pairs = np.array([[0, 2], [1, 2]])
neighbor_phasings = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                              [[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=int)
error_rate = 0.02
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Valid: the same candidate set scored on the likelihood alone, which
        #     separates the three terms of the combination ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
import itertools
base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
cols = [np.array(p, dtype=int) for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
candidates = np.array([np.concatenate([base, c[:, None]], axis=1) for c in cols], dtype=int)
positions = np.array([0, 1, 2])
reads = np.array([[0, 0, 1, -1],
                  [0, 1, 0, -1],
                  [1, 0, 1, -1],
                  [1, 1, 0, -1],
                  [-1, 0, 1, 0],
                  [-1, 1, 0, 1]], dtype=int)
neighbor_pairs = np.array([[0, 2], [1, 2]])
neighbor_phasings = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                              [[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=int)
error_rate = 0.02
weights = np.array([1.0, 0.0, 0.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Valid: a neighbour vertex reaching outside the window, whose
        #     decoded phasing cannot be checked and must not be counted ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
import itertools
base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
cols = [np.array(p, dtype=int) for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
candidates = np.array([np.concatenate([base, c[:, None]], axis=1) for c in cols], dtype=int)
positions = np.array([0, 1, 2])
reads = np.array([[0, 0, 1, -1],
                  [0, 1, 0, -1],
                  [1, 0, 1, -1],
                  [1, 1, 0, -1],
                  [-1, 0, 1, 0],
                  [-1, 1, 0, 1]], dtype=int)
neighbor_pairs = np.array([[0, 2], [2, 3]])
neighbor_phasings = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]],
                              [[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=int)
error_rate = 0.02
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Boundary: a single candidate, for which every rescaled term
        #     collapses and the score must be exactly zero ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
candidates = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=int)
positions = np.array([0, 1])
reads = np.array([[0, 1, -1], [1, 0, -1], [1, 1, 0]], dtype=int)
neighbor_pairs = np.array([[0, 1]])
neighbor_phasings = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=int)
error_rate = 0.02
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Boundary: no decoded neighbours at all, so the agreement term is
        #     flat and only likelihood and error correction separate candidates ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
import itertools
base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
cols = [np.array(p, dtype=int) for p in itertools.product((0, 1), repeat=4) if sum(p) == 2]
candidates = np.array([np.concatenate([base, c[:, None]], axis=1) for c in cols], dtype=int)
positions = np.array([0, 1, 2])
reads = np.array([[0, 0, 1, -1],
                  [0, 1, 0, -1],
                  [1, 0, 1, -1],
                  [1, 1, 0, -1]], dtype=int)
neighbor_pairs = np.zeros((0, 2), dtype=int)
neighbor_phasings = np.zeros((0, 4, 2), dtype=int)
error_rate = 0.02
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Edge: fragments that touch only part of the window, so the
        #     likelihood exponent must follow each fragment's own coverage ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
import itertools
base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
cols = [np.array(p, dtype=int) for p in itertools.product((0, 1), repeat=4) if sum(p) == 3]
candidates = np.array([np.concatenate([base, c[:, None]], axis=1) for c in cols], dtype=int)
positions = np.array([0, 2, 4])
reads = np.array([[0, -1, -1, -1, 1],
                  [-1, -1, 1, -1, -1],
                  [1, -1, 0, -1, 1],
                  [-1, -1, -1, -1, 0],
                  [0, -1, 1, -1, -1]], dtype=int)
neighbor_pairs = np.array([[0, 4]])
neighbor_phasings = np.array([[[0, 1], [0, 1], [1, 0], [1, 1]]], dtype=int)
error_rate = 0.1
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
""",
            "call": "sig(score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
            "gold_call": "sig(_oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, error_rate, weights), 1.0)",
        },
        # --- Invalid: a weight vector of the wrong length ---
        {
            "setup": """import numpy as np
candidates = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 1])
reads = np.array([[0, 1], [1, 0]], dtype=int)
neighbor_pairs = np.zeros((0, 2), dtype=int)
neighbor_phasings = np.zeros((0, 2, 2), dtype=int)
weights = np.array([0.5, 0.5])
def run_model():
    try:
        score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an error rate of exactly zero ---
        {
            "setup": """import numpy as np
candidates = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 1])
reads = np.array([[0, 1], [1, 0]], dtype=int)
neighbor_pairs = np.zeros((0, 2), dtype=int)
neighbor_phasings = np.zeros((0, 2, 2), dtype=int)
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
def run_model():
    try:
        score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.0, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.0, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a decoded neighbour phasing of the wrong ploidy ---
        {
            "setup": """import numpy as np
candidates = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]], dtype=int)
positions = np.array([0, 1])
reads = np.array([[0, 1], [1, 0]], dtype=int)
neighbor_pairs = np.array([[0, 1]])
neighbor_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
def run_model():
    try:
        score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a neighbour vertex naming a SNP outside the fragments ---
        {
            "setup": """import numpy as np
candidates = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 1])
reads = np.array([[0, 1], [1, 0]], dtype=int)
neighbor_pairs = np.array([[0, 9]])
neighbor_phasings = np.array([[[0, 1], [1, 0]]], dtype=int)
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
def run_model():
    try:
        score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a window position outside the fragment matrix ---
        {
            "setup": """import numpy as np
candidates = np.array([[[0, 1], [1, 0]]], dtype=int)
positions = np.array([0, 6])
reads = np.array([[0, 1], [1, 0]], dtype=int)
neighbor_pairs = np.zeros((0, 2), dtype=int)
neighbor_phasings = np.zeros((0, 2, 2), dtype=int)
weights = np.array([1.0 / 12.0, 10.0 / 12.0, 1.0 / 12.0])
def run_model():
    try:
        score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_score_phasing_candidates(candidates, positions, reads, neighbor_pairs, neighbor_phasings, 0.02, weights)
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
