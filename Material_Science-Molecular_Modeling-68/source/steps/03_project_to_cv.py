"""
Implement project_to_cv, which projects one or more system-averaged descriptors into the reduced collective-variable (CV) space defined by a PCA mean and basis. This is applied both to the current configuration's descriptor and to the full reference set, since the projected reference points later serve as kernel centers for the density estimate.

Once the PCA mean and basis are known from the reference set, any descriptor - the current configuration's  or  a reference configuration's - is projected into the same low-dimensional CV space by centering and applying the truncated basis, so that all quantities compared later (kernel evaluations, density estimates) live in a consistent, shared coordinate system.

Returns
-------
np.ndarray of shape (N, k), the projected collective-variable coordinates, as a NumPy float array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_to_cv(S: np.ndarray, mu: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    '''Project one or more descriptors into the reduced CV space.

    Parameters
    ----------
    S : np.ndarray
        Array of shape (N, D), one or more descriptor vectors to project.
    mu : np.ndarray
        Array of shape (D,), the PCA mean (from compute_pca_basis).
    V_k : np.ndarray
        Array of shape (D, k), the PCA projection basis (from compute_pca_basis).

    Returns
    -------
    S_cv : np.ndarray
        Array of shape (N, k), the projected collective-variable coordinates.

    Raises
    ------
    ValueError
        If S is not 2D, if mu is not 1D, if V_k is not 2D, if S.shape[1] !=
        mu.shape[0], if mu.shape[0] != V_k.shape[0], or if any input
        contains non-finite values.
    '''
    return S_cv  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_project_to_cv(S: np.ndarray, mu: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    S = np.asarray(S, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V_k = np.asarray(V_k, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be a 2D array of shape (N, D)")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array of shape (D,)")
    if V_k.ndim != 2:
        raise ValueError("V_k must be a 2D array of shape (D, k)")
    if S.shape[1] != mu.shape[0] or mu.shape[0] != V_k.shape[0]:
        raise ValueError("S, mu, and V_k must have consistent descriptor dimension D")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(V_k))):
        raise ValueError("S, mu, and V_k must contain only finite values")
    return ((S - mu) @ V_k).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: project the full reference set (seed=7) with k=3
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(7)\n"
                "S_ref = rng.standard_normal((20, 12))\n"
                "mu = S_ref.mean(axis=0)\n"
                "Shat = S_ref - mu\n"
                "_, _, Vt = np.linalg.svd(Shat, full_matrices=False)\n"
                "V_k = Vt.T[:, :3]"
            ),
            "call": "project_to_cv(S_ref, mu, V_k)",
            "gold_call": "_oracle_project_to_cv(S_ref, mu, V_k)",
        },
        {
            # Boundary case: single-row input (N=1), current configuration's descriptor
            "setup": (
                "import numpy as np\n"
                "S = np.array([[0.5, -0.2]])\n"
                "mu = np.array([0.0, 0.0])\n"
                "V_k = np.array([[1.0], [0.0]])"
            ),
            "call": "project_to_cv(S, mu, V_k)",
            "gold_call": "_oracle_project_to_cv(S, mu, V_k)",
        },
        {
            # Edge case: k equals D, mean exactly cancels one row
            "setup": (
                "import numpy as np\n"
                "S = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])\n"
                "mu = S.mean(axis=0)\n"
                "V_k = np.array([[1.0, 0.0], [0.0, 1.0]])"
            ),
            "call": "project_to_cv(S, mu, V_k)",
            "gold_call": "_oracle_project_to_cv(S, mu, V_k)",
        },
        {
            # Invalid-input case: dimension mismatch should raise ValueError
            "setup": (
        "import numpy as np\n"
        "S = np.array([[1.0, 2.0, 3.0]])\n"
        "mu = np.array([0.0, 0.0])\n"
        "V_k = np.array([[1.0], [0.0]])\n"
        "def run_model():\n"
        "    try:\n"
        "        project_to_cv(S, mu, V_k)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_project_to_cv(S, mu, V_k)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2"
    ),
    "call": "run_model()",
    "gold_call": "run_gold()",
},
    ]
