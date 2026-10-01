"""
At the physical level, discard the boundary pair to obtain the density matrix of the edge
register, whose spectrum is the entanglement spectrum of the half chain. Returns the
register density matrix.

R has legs [k0, k1, e, b0, b1, f] with the register dimension d = 2**T after T layers. The
boundary sites (0, 1) belong to A and are traced: rho[e, f] = sum over a, b of R[a, b, e, a,
b, f]. Return the d x d float array.

Returns
-------
np.ndarray: d x d float array, the edge-register density matrix.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def edge_reduced_state(R: "np.ndarray") -> "np.ndarray":
    """At the physical level, discard the boundary pair to obtain the density matrix of the
    edge register, whose spectrum is the entanglement spectrum of the half chain. Returns
    the register density matrix.

    Args:
        R: float array of shape (2, 2, d, 2, 2, d) at the physical level.

    Returns:
        np.ndarray: d x d float array, the edge-register density matrix.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _edge_reduced_state(R):
    return np.einsum("abeabf->ef", np.asarray(R, dtype=float))


def _oracle_edge_reduced_state(R: "np.ndarray") -> "np.ndarray":
    return _edge_reduced_state(R)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(8); R = rng.normal(size=(2, 2, 8, 2, 2, 8))',
            'call': 'edge_reduced_state(*copy.deepcopy((R,)))',
            'gold_call': '_oracle_edge_reduced_state(*copy.deepcopy((R,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(18); R = rng.normal(size=(2, 2, 2, 2, 2, 2))',
            'call': 'edge_reduced_state(*copy.deepcopy((R,)))',
            'gold_call': '_oracle_edge_reduced_state(*copy.deepcopy((R,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; R = np.zeros((2, 2, 2, 2, 2, 2)); R[0, 1, 0, 1, 0, 1] = 0.4; R[0, 1, 1, 0, 1, 1] = 9.0; R[1, 1, 1, 1, 1, 0] = 0.6',
            'call': 'edge_reduced_state(*copy.deepcopy((R,)))',
            'gold_call': '_oracle_edge_reduced_state(*copy.deepcopy((R,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(28); R = rng.normal(size=(2, 2, 1, 2, 2, 1))',
            'call': 'edge_reduced_state(*copy.deepcopy((R,)))',
            'gold_call': '_oracle_edge_reduced_state(*copy.deepcopy((R,)))',
        },
    ]
