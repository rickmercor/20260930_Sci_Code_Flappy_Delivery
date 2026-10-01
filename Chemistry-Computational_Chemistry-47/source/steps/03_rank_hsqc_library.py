"""
Return the source-defined direct ordering and aligned spectral signal in original candidate order.

Preserve candidate identity throughout the ranking. The pinned implementation defines the spectral transformation and stable ordering conventions.

Returns
-------
A zero-based integer order and the aligned source-defined spectral-signal vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rank_hsqc_library(
    distances: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the source-defined direct ordering and aligned spectral signal.

    Preserve candidate identity and keep the returned signal in original
    candidate order. Follow the pinned transformation and ordering conventions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rank_hsqc_library(
    distances: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    import numpy as np

    values = np.asarray(distances, dtype=float)

    if (
        values.ndim != 1
        or len(values) == 0
        or np.any(values < 0)
    ):
        raise ValueError("nonempty nonnegative vector required")

    indices = np.arange(len(values), dtype=int)
    order = np.lexsort((indices, values))
    similarities = 1.0 / (1.0 + values)

    return order.astype(int), similarities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(value):\n'
               '    order, similarities = value\n'
               '    return np.concatenate((\n'
               '        order.astype(float),\n'
               '        similarities,\n'
               '    ))\n'
               'd=np.array([2.,1.,1.])',
      'call': 'pack(rank_hsqc_library(copy.deepcopy(d)))',
      'gold_call': 'pack(_oracle_rank_hsqc_library(copy.deepcopy(d)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(value):\n'
               '    order, similarities = value\n'
               '    return np.concatenate((\n'
               '        order.astype(float),\n'
               '        similarities,\n'
               '    ))\n'
               'd=np.array([0.,4.])',
      'call': 'pack(rank_hsqc_library(copy.deepcopy(d)))',
      'gold_call': 'pack(_oracle_rank_hsqc_library(copy.deepcopy(d)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(value):\n'
               '    order, similarities = value\n'
               '    return np.concatenate((\n'
               '        order.astype(float),\n'
               '        similarities,\n'
               '    ))\n'
               'd=np.array([37.1,3.2,10.9,3.2])',
      'call': 'pack(rank_hsqc_library(copy.deepcopy(d)))',
      'gold_call': 'pack(_oracle_rank_hsqc_library(copy.deepcopy(d)))'}]
