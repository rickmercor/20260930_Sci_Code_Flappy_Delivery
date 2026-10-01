"""
Combine spectral-distance and network evidence under the source-defined correction.

The two signals are transformed and normalized independently before the supplied weights are applied. Follow the pinned implementation for degenerate columns and weight handling.

Returns
-------
The composite-score vector and the normalized network-signal vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def network_corrected_scores(
    query_distances: "np.ndarray",
    pwra_scores: "np.ndarray",
    distance_weight: float,
    network_weight: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the source-defined composite and normalized network signal.

    The two input vectors remain in original candidate order. Apply the
    supplied nonnegative evidence weights using the pinned conventions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_network_corrected_scores(
    query_distances: "np.ndarray",
    pwra_scores: "np.ndarray",
    distance_weight: float,
    network_weight: float,
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    distances = np.asarray(
        query_distances,
        dtype=float,
    )
    pwra = np.asarray(
        pwra_scores,
        dtype=float,
    )

    if (
        distances.ndim != 1
        or distances.shape != pwra.shape
        or len(distances) == 0
    ):
        raise ValueError(
            "equal nonempty vectors required"
        )

    if (
        min(distance_weight, network_weight) < 0
        or distance_weight + network_weight <= 0
    ):
        raise ValueError("invalid weights")

    total = float(
        distance_weight + network_weight
    )
    distance_weight /= total
    network_weight /= total

    inverse = 1.0 / (1.0 + distances)

    pwra_span = float(np.max(pwra) - np.min(pwra))
    if pwra_span == 0:
        normalized_pwra = np.ones_like(pwra)
    else:
        normalized_pwra = (pwra - np.min(pwra)) / pwra_span

    inverse_span = float(np.max(inverse) - np.min(inverse))
    if inverse_span == 0:
        normalized_inverse = np.ones_like(inverse)
    else:
        normalized_inverse = (
            inverse - np.min(inverse)
        ) / inverse_span

    combined = (
        distance_weight * normalized_inverse
        + network_weight * normalized_pwra
    )

    return combined, normalized_pwra

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' combined,normalized=x; return np.concatenate((combined,normalized))\n'
               'd=np.array([0.,3.,15.,1.]); p=np.array([100.,0.,20.,50.])',
      'call': 'pack(network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
              'copy.deepcopy(0.8), copy.deepcopy(0.2)))',
      'gold_call': 'pack(_oracle_network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
                   'copy.deepcopy(0.8), copy.deepcopy(0.2)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' combined,normalized=x; return np.concatenate((combined,normalized))\n'
               'd=np.array([2.,2.,2.]); p=np.array([5.,7.,9.])',
      'call': 'pack(network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
              'copy.deepcopy(0.8), copy.deepcopy(0.2)))',
      'gold_call': 'pack(_oracle_network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
                   'copy.deepcopy(0.8), copy.deepcopy(0.2)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' combined,normalized=x; return np.concatenate((combined,normalized))\n'
               'd=np.array([0.,1.]); p=np.array([0.,1.])',
      'call': 'pack(network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
              'copy.deepcopy(4), copy.deepcopy(1)))',
      'gold_call': 'pack(_oracle_network_corrected_scores(copy.deepcopy(d), copy.deepcopy(p), '
                   'copy.deepcopy(4), copy.deepcopy(1)))'}]
