"""
Generate the ordered roots-of-unity sample points.

A Fourier sampling grid contains $S=k+1$ unit-circle points, ordered as

$x_j=\omega^{-j}$ for $j=0,\ldots,k$, where $\omega=\exp(2\pi i/S)$.

The ordering fixes the transform sign used for coefficient recovery.

Returns
-------
A complex NumPy array of shape $(k+1,)$, where $k$ is `determinant_degree`, in increasing sample index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_unit_circle_samples(determinant_degree: int) -> np.ndarray:
    r"""Return $x_j=\omega^{-j}$ for $j=0,\ldots,k$.
    
    Parameters
    ----------
    determinant_degree : int
        Integer $k\ge1$ specifying a grid of $S=k+1$ points.
    
    Returns
    -------
    sample_points : np.ndarray
        Complex vector of shape $(k+1,)$ in increasing sample index,
        with $x_j=\exp(-2\pi i j/(k+1))$ and $x_0=1$.
    
    Raises
    ------
    ValueError
        If `determinant_degree` is not an integer or is less than one.
    
    Notes
    -----
    The negative exponential sign pairs these samples with an inverse
    Fourier transform for recovering ascending polynomial coefficients.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_generate_unit_circle_samples(determinant_degree: int) -> np.ndarray:
    """Reference roots-of-unity construction."""
    if not isinstance(determinant_degree, (int, np.integer)) or determinant_degree < 1:
        raise ValueError("determinant_degree must be a positive integer")
    count = int(determinant_degree) + 1
    return np.exp(-2j * np.pi * np.arange(count) / count)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return task-scale, smallest-valid, and invalid-degree cases."""
    return [
        {
            "setup": "determinant_degree = 10",
            "call": "generate_unit_circle_samples(determinant_degree)",
            "gold_call": "_oracle_generate_unit_circle_samples(determinant_degree)",
        },
        {
            "setup": "determinant_degree = 1",
            "call": "generate_unit_circle_samples(determinant_degree)",
            "gold_call": "_oracle_generate_unit_circle_samples(determinant_degree)",
        },
        {
            "setup": "determinant_degree = 0\ndef run_model():\n    try:\n        generate_unit_circle_samples(determinant_degree)\n        return 0\n    except ValueError:\n        return 1\ndef run_gold():\n    try:\n        _oracle_generate_unit_circle_samples(determinant_degree)\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
