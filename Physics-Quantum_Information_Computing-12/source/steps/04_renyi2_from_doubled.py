"""
Close the doubled boundary operator at the physical level and convert the resulting purity
into the second Renyi entanglement entropy in bits. Returns S2.

Q has legs [k1_0, k1_1, b1_0, b1_1, k2_0, k2_1, b2_0, b2_1]. Everything that belongs to B
has already been contracted across the copies on the way down, so at the physical level the
boundary sites (0, 1) are traced inside each copy separately: the purity is P = sum over a,
b, c, d of Q[a, b, a, b, c, d, c, d]. Return S2 = -log2(P) as a Python float.

Returns
-------
float: second Renyi entropy S2 = -log2(purity), in bits.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def renyi2_from_doubled(Q: "np.ndarray") -> float:
    """Close the doubled boundary operator at the physical level and convert the resulting
    purity into the second Renyi entanglement entropy in bits. Returns S2.

    Args:
        Q: float array of shape (2,)*8 at the physical level.

    Returns:
        float: second Renyi entropy S2 = -log2(purity), in bits.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _renyi2_from_doubled(Q):
    return float(-np.log2(np.einsum("ababcdcd->", np.asarray(Q, dtype=float))))


def _oracle_renyi2_from_doubled(Q: "np.ndarray") -> float:
    return _renyi2_from_doubled(Q)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(2); Q = np.abs(rng.normal(size=[2] * 8))',
            'call': 'renyi2_from_doubled(*copy.deepcopy((Q,)))',
            'gold_call': '_oracle_renyi2_from_doubled(*copy.deepcopy((Q,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(4); Q = 0.03 * np.abs(rng.normal(size=[2] * 8))',
            'call': 'renyi2_from_doubled(*copy.deepcopy((Q,)))',
            'gold_call': '_oracle_renyi2_from_doubled(*copy.deepcopy((Q,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; Q = np.zeros([2] * 8); Q[0, 1, 0, 1, 1, 0, 1, 0] = 0.3; Q[1, 1, 1, 1, 0, 0, 0, 0] = 0.2',
            'call': 'renyi2_from_doubled(*copy.deepcopy((Q,)))',
            'gold_call': '_oracle_renyi2_from_doubled(*copy.deepcopy((Q,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; Q = np.full([2] * 8, 0.01); Q[0, 0, 1, 1, 0, 0, 0, 0] = 5.0; Q[0, 0, 0, 0, 0, 1, 1, 0] = 7.0',
            'call': 'renyi2_from_doubled(*copy.deepcopy((Q,)))',
            'gold_call': '_oracle_renyi2_from_doubled(*copy.deepcopy((Q,)))',
        },
    ]
