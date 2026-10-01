"""
Extract the second-smallest positive eigenvalue from an ordered discrete spectrum.

The paper defines the discrete spectrum as an increasing sequence of positive eigenvalues, repeated according to multiplicity. The requested benchmark quantity is the second member of that ordered positive sequence.

Returns
-------
float, the second-smallest positive eigenvalue
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_second_eigenvalue(eigenvalues: np.ndarray) -> float:
    """
    Extract the second-smallest positive eigenvalue.

    Parameters
    ----------
    eigenvalues : np.ndarray
        One-dimensional array of eigenvalues.

    Returns
    -------
    result : float
        Second-smallest positive eigenvalue.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_extract_second_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Reference implementation."""
    x = np.asarray(eigenvalues, dtype=float)

    if x.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("eigenvalues must be finite")

    positive = np.sort(x[x > 0.0])

    if len(positive) < 2:
        raise ValueError("at least two positive eigenvalues are required")

    return float(positive[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
x = np.array([92.0, 52.0, 128.0, 92.2])""",
            "call": "extract_second_eigenvalue(x)",
            "gold_call": "_oracle_extract_second_eigenvalue(x)",
        },
        {
            "setup": """import numpy as np
x = np.array([1.0, 1.0, 2.0])""",
            "call": "extract_second_eigenvalue(x)",
            "gold_call": "_oracle_extract_second_eigenvalue(x)",
        },
        {
            "setup": """import numpy as np
x = np.array([-5.0, 0.0, 3.0, 7.0])""",
            "call": "extract_second_eigenvalue(x)",
            "gold_call": "_oracle_extract_second_eigenvalue(x)",
        },
    ]
