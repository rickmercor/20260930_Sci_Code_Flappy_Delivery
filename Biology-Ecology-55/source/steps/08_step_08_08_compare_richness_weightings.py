"""
Conditioned on systems with S_p > 0, the system-centered richness probability is the mean of F_p^k across systems, while the state-centered probability pools all states before normalizing. These weight systems differently whenever multiplicity covaries with within-system richness composition. The signed pooling bias at richness k is p_state^k - p_system^k, positive when high-multiplicity systems disproportionately contribute states of that richness.

Returns
-------
Vector [p_system^k, p_state^k, p_state^k - p_system^k].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compare_richness_weightings(
    system_summaries: np.ndarray,
    probability_vectors: np.ndarray,
    target_richness: int,
) -> np.ndarray:
    '''Compare system-centered and pooled-state richness probabilities.

    Parameters
    ----------
    system_summaries : np.ndarray
        Nonnegative integer rows [S_p, S_p^0, ..., S_p^N].
    probability_vectors : np.ndarray
        Probability rows [F_p^empty, F_p^0, ..., F_p^N] derived from the
        supplied summaries.
    target_richness : int
        Richness k between zero and N, inclusive.

    Raises
    ------
    ValueError
        If inputs are inconsistent, target_richness is outside 0 through N,
        probability rows are invalid, or no system has a stable state.

    Returns
    -------
    comparison : np.ndarray
        Float vector [p_system^k, p_state^k, pooling_bias].
    '''
    return comparison  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compare_richness_weightings(
    system_summaries: np.ndarray,
    probability_vectors: np.ndarray,
    target_richness: int,
) -> np.ndarray:
    """Reference implementation."""
    system_summaries = np.asarray(system_summaries)
    probability_vectors = np.asarray(probability_vectors, dtype=float)
    if system_summaries.ndim != 2 or system_summaries.shape[0] == 0:
        raise ValueError("system_summaries must be a nonempty matrix")
    if system_summaries.shape[1] < 3:
        raise ValueError("system_summaries must include S_p and richness counts")
    if probability_vectors.shape != system_summaries.shape:
        raise ValueError("probability_vectors must match system_summaries")
    if not np.all(np.isfinite(system_summaries)) or not np.all(
        system_summaries == np.floor(system_summaries)
    ):
        raise ValueError("system_summaries must contain finite integers")
    if np.any(system_summaries < 0):
        raise ValueError("system_summaries must be nonnegative")
    system_summaries = system_summaries.astype(int)
    multiplicities = system_summaries[:, 0]
    counts = system_summaries[:, 1:]
    if not np.array_equal(multiplicities, np.sum(counts, axis=1)):
        raise ValueError("S_p must equal the sum of richness counts")
    n_species = system_summaries.shape[1] - 2
    if isinstance(target_richness, (bool, np.bool_)) or not isinstance(
        target_richness, (int, np.integer)
    ):
        raise ValueError("target_richness must be an integer from zero to N")
    target_richness = int(target_richness)
    if target_richness < 0 or target_richness > n_species:
        raise ValueError("target_richness must be an integer from zero to N")
    if not np.all(np.isfinite(probability_vectors)) or np.any(probability_vectors < 0.0):
        raise ValueError("probability_vectors must be finite and nonnegative")
    if not np.allclose(np.sum(probability_vectors, axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("each probability vector must sum to one")

    expected = np.zeros(probability_vectors.shape, dtype=float)
    zero_state = multiplicities == 0
    expected[zero_state, 0] = 1.0
    positive = ~zero_state
    expected[positive, 1:] = counts[positive] / multiplicities[positive, None]
    if not np.allclose(probability_vectors, expected, atol=1e-12, rtol=0.0):
        raise ValueError("probability_vectors are inconsistent with system_summaries")
    if not np.any(positive):
        raise ValueError("at least one system must contain a stable state")

    richness_column = target_richness + 1
    system_centered = float(np.mean(probability_vectors[positive, richness_column]))
    state_centered = float(
        np.sum(system_summaries[positive, richness_column]) / np.sum(multiplicities[positive])
    )
    return np.array(
        [system_centered, state_centered, state_centered - system_centered],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''system_summaries = np.array(
    [
        [3, 0, 2, 1, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [4, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [2, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
    ],
    dtype=int,
)
probability_vectors = np.array(
    [
        [0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)
target_richness = 2
''',
            "call": "compare_richness_weightings(system_summaries, probability_vectors, target_richness).tolist()",
            "gold_call": "_oracle_compare_richness_weightings(system_summaries, probability_vectors, target_richness).tolist()",
        },
        {
            "setup": '''system_summaries = np.array([[1, 0, 1], [1, 0, 1]], dtype=int)
probability_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
target_richness = 1
''',
            "call": "compare_richness_weightings(system_summaries, probability_vectors, target_richness).tolist()",
            "gold_call": "_oracle_compare_richness_weightings(system_summaries, probability_vectors, target_richness).tolist()",
        },
        {
            "setup": '''system_summaries = np.array([[1, 0, 1]], dtype=int)
probability_vectors = np.array([[0.0, 0.0, 1.0]])
target_richness = 2
def run_model():
    try:
        compare_richness_weightings(system_summaries, probability_vectors, target_richness)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compare_richness_weightings(system_summaries, probability_vectors, target_richness)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
''',
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
