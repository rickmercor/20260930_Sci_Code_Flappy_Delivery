"""
Build a positive definite matrix A from A2's eigenvector structure, with nine prescribed eigenvalues logarithmically spaced from 1 to condition_number (condition_number^(i/8) for i = 0, ..., 8).

A2 is generally indefinite, so it cannot serve directly as the positive definite matrix "A" needed for the pencil matrix-function evaluation. A must instead be built deterministically from A2 so that A shares A2's own orthonormal eigenvectors exactly, with A's prescribed eigenvalues assigned to them in the same relative order as A2's own eigenvalues (A2's smallest-eigenvalue eigenvector receives A's smallest prescribed eigenvalue, ..., A2's largest-eigenvalue eigenvector receives A's largest prescribed eigenvalue condition_number).

Returns
-------
A 9x9 real symmetric positive definite matrix with eigenvalues exactly condition_number^(i/8) for i = 0, ..., 8.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_conditioned_pencil(A2: "np.ndarray", condition_number: float) -> "np.ndarray":
    """Build a positive definite, severely ill-conditioned matrix A from
    A2, with nine prescribed eigenvalues logarithmically spaced from 1 to
    condition_number (condition_number^(i/8) for i = 0, ..., 8).

    Parameters
    ----------
    A2 : np.ndarray
        (9, 9) real symmetric matrix.
    condition_number : float
        The desired ratio of largest to smallest eigenvalue of A (> 1).

    Returns
    -------
    A : np.ndarray
        (9, 9) real symmetric positive definite matrix that shares A2's
        own orthonormal eigenvectors exactly, with eigenvalues exactly
        condition_number^(i/8) for i = 0, ..., 8 assigned to them in the
        same relative order as A2's own eigenvalues (A2's
        smallest-eigenvalue eigenvector receives A's smallest prescribed
        eigenvalue, and so on up to A2's largest-eigenvalue eigenvector
        receiving A's largest prescribed eigenvalue condition_number).

    Raises
    ------
    ValueError
        If A2 is not a 9x9 symmetric matrix, or if condition_number is
        not strictly greater than 1.
    """
    return A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_conditioned_pencil(A2: "np.ndarray", condition_number: float) -> "np.ndarray":
    import numpy as np
    A2 = np.asarray(A2, dtype=float)
    if A2.shape[0] != A2.shape[1] or A2.shape[0] != 9:
        raise ValueError("A2 must be a 9x9 matrix")
    if not np.allclose(A2, A2.T, atol=1e-8):
        raise ValueError("A2 must be symmetric")
    if condition_number <= 1.0:
        raise ValueError("condition_number must be > 1")
    n = A2.shape[0]
    evals, evecs = np.linalg.eigh(A2)  # ascending order
    for k in range(evecs.shape[1]):
        col = evecs[:, k]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            evecs[:, k] = -col
    prescribed = 10.0 ** np.linspace(0.0, np.log10(condition_number), n)
    A = evecs @ np.diag(prescribed) @ evecs.T
    A = 0.5 * (A + A.T)
    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    import numpy as np
    return [
        # --- Normal case: the actual task instance's A2, condition 1e8 ---
        {
            "setup": """import numpy as np
A2 = np.array([
    [16.5778034479819, -5.5283307619849, 1.4953036216683, -3.5207229860391, 6.2616029109144, -3.537079386105, -3.0827491183509, 1.2862564132768, -5.8870301513278],
    [-5.5283307619849, 0.328913824155, 2.8762807423355, 2.1237343354472, 0.4671253955339, 0.7467785546017, 1.4581360651548, 1.718312874976, -2.4674916086529],
    [1.4953036216683, 2.8762807423355, -2.9754623145709, 0.2985143361469, -2.394514673282, 0.2906434717331, 6.4186133605821, -0.0103306683558, 5.5946981477741],
    [-3.5207229860391, 2.1237343354472, 0.2985143361469, 0.8936808218989, -0.9327967300772, 1.0818267213255, 1.9858863587058, -0.6791582415432, -1.6731779149174],
    [6.2616029109144, 0.4671253955339, -2.394514673282, -0.9327967300772, 0.3418010425452, -1.6018105202283, 1.1940734794077, 4.1178098778913, 3.3200622344099],
    [-3.537079386105, 0.7467785546017, 0.2906434717331, 1.0818267213255, -1.6018105202283, -3.2109110632581, 0.7185105391816, 0.2836743166271, 1.6802828874668],
    [-3.0827491183509, 1.4581360651548, 6.4186133605821, 1.9858863587058, 1.1940734794077, 0.7185105391816, -4.4768447441237, 3.2449278377128, -0.3285811550915],
    [1.2862564132768, 1.718312874976, -0.0103306683558, -0.6791582415432, 4.1178098778913, 0.2836743166271, 3.2449278377128, -1.1240314581512, -1.3680784169686],
    [-5.8870301513278, -2.4674916086529, 5.5946981477741, -1.6731779149174, 3.3200622344099, 1.6802828874668, -0.3285811550915, -1.3680784169686, -4.1535075171468],
])
condition_number = 1e8
A2_ref = A2.copy()
""",
            "call": "build_conditioned_pencil(A2, condition_number)",
            "gold_call": "_oracle_build_conditioned_pencil(A2_ref, condition_number)",
            "tol": 10.0,
        },
        # --- Boundary case: A2 already SPD, small condition number ---
        {
            "setup": """import numpy as np
A2 = np.diag([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])
condition_number = 2.0
A2_ref = A2.copy()
""",
            "call": "build_conditioned_pencil(A2, condition_number)",
            "gold_call": "_oracle_build_conditioned_pencil(A2_ref, condition_number)",
            "tol": 1e-9,
        },
        # --- Edge case: condition_number <= 1 should raise ---
        {
            "setup": """import numpy as np
A2 = np.diag([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])
condition_number = 0.5
A2_ref = A2.copy()
def run_model():
    try:
        build_conditioned_pencil(A2, condition_number)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_conditioned_pencil(A2_ref, condition_number)
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
