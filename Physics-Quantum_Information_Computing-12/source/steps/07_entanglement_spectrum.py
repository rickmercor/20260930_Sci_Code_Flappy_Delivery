"""
Diagonalise a real symmetric density matrix and return its eigenvalues in descending order,
the entanglement spectrum lambda_0 >= lambda_1 >= ... .

The register density matrix is real symmetric. Return all eigenvalues, largest first, as a
float numpy array; do not clip, renormalise or drop small or degenerate values.

Returns
-------
np.ndarray: float array of the eigenvalues in descending order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def entanglement_spectrum(rho: "np.ndarray") -> "np.ndarray":
    """Diagonalise a real symmetric density matrix and return its eigenvalues in descending
    order, the entanglement spectrum lambda_0 >= lambda_1 >= ... .

    Args:
        rho: real symmetric d x d matrix.

    Returns:
        np.ndarray: float array of the eigenvalues in descending order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _entanglement_spectrum(rho):
    m = np.asarray(rho, dtype=float)
    return np.linalg.eigvalsh(0.5 * (m + m.T))[::-1].copy()


def _oracle_entanglement_spectrum(rho: "np.ndarray") -> "np.ndarray":
    return _entanglement_spectrum(rho)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(1); A = rng.normal(size=(8, 8)); rho = A @ A.T / np.trace(A @ A.T)',
            'call': 'entanglement_spectrum(*copy.deepcopy((rho,)))',
            'gold_call': '_oracle_entanglement_spectrum(*copy.deepcopy((rho,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rng = np.random.default_rng(7); A = rng.normal(size=(16, 16)); rho = A @ A.T / np.trace(A @ A.T)',
            'call': 'entanglement_spectrum(*copy.deepcopy((rho,)))',
            'gold_call': '_oracle_entanglement_spectrum(*copy.deepcopy((rho,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rho = np.diag([0.1, 0.6, 0.2, 0.1])',
            'call': 'entanglement_spectrum(*copy.deepcopy((rho,)))',
            'gold_call': '_oracle_entanglement_spectrum(*copy.deepcopy((rho,)))',
        },
        {
            'setup': 'import numpy as np\nimport copy\nimport numpy as np; rho = np.array([[0.55, 0.4], [0.4, 0.45]])',
            'call': 'entanglement_spectrum(*copy.deepcopy((rho,)))',
            'gold_call': '_oracle_entanglement_spectrum(*copy.deepcopy((rho,)))',
        },
    ]
