"""
Build the Fourier-space linear operator for every closure row.

Combining the heat-flux closure with Poisson coupling produces a 3-by-3 mode generator.

Returns
-------
np.ndarray: complex array of shape (m, 3, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def assemble_moment_generators(closure_table: np.ndarray) -> np.ndarray:
    """Return one complex ``3 x 3`` generator per closure row.

    Parameters
    ----------
    closure_table : np.ndarray
        Rows containing ``kappa, Re(zeta), Im(zeta), Q1, Q3``.

    Returns
    -------
    np.ndarray
        Complex array with shape ``(m, 3, 3)`` for state ``[n,u,p]``.

    Raises
    ------
    ValueError
        If the closure table is invalid or contains nonpositive ``kappa``, ``Q1``, or ``Q3``.
    """
    return np.empty((np.asarray(closure_table).shape[0], 3, 3), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_moment_generators(closure_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    closure = np.asarray(closure_table, dtype=float)
    if closure.ndim != 2 or closure.shape[1] != 5 or closure.shape[0] == 0:
        raise ValueError("closure_table must have shape (m, 5)")
    if not np.all(np.isfinite(closure)):
        raise ValueError("closure_table must be finite")
    matrices = np.zeros((closure.shape[0], 3, 3), dtype=complex)
    sqrt_two = np.sqrt(2.0)
    for row, values in enumerate(closure):
        kappa, _, _, q1, q3 = values
        if kappa <= 0.0 or q1 <= 0.0 or q3 <= 0.0:
            raise ValueError("kappa, Q1, and Q3 must be positive")
        matrices[row, 0, 1] = -1j * sqrt_two * kappa
        matrices[row, 1, 0] = -1j / (sqrt_two * kappa)
        matrices[row, 1, 2] = -1j * kappa / sqrt_two
        matrices[row, 2, 0] = sqrt_two * kappa * q3
        matrices[row, 2, 1] = -1j * sqrt_two * kappa * (3.0 + q1)
        matrices[row, 2, 2] = -sqrt_two * kappa * q3
    if not np.all(np.isfinite(matrices)):
        raise ValueError("moment generators must be finite")
    return matrices

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Compare every generator entry and check the invalid-input contract."""
    return [
        {
            "setup": "import numpy as np; closure = np.array([[0.4, 2.2, -0.1, 1.35, 0.78], [0.6, 1.8, -0.3, 2.0, 1.4]])",
            "call": "assemble_moment_generators(closure)",
            "gold_call": "_oracle_assemble_moment_generators(closure)",
        },
        {
            "setup": "import numpy as np; closure = np.array([[0.3, 3.0, -0.01, 0.7, 0.2]])",
            "call": "assemble_moment_generators(closure)",
            "gold_call": "_oracle_assemble_moment_generators(closure)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "closure = np.array([[0.4, 2.0, -0.1, -1.0, 0.8]])\n"
                "def status(fn):\n"
                "    try:\n"
                "        fn()\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "status(lambda: assemble_moment_generators(closure))",
            "gold_call": "status(lambda: _oracle_assemble_moment_generators(closure))",
        },
    ]
