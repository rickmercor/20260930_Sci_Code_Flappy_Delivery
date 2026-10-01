"""
Establish whether the dominant kinetic eigenmode is observable in T₁ after S₁-only excitation. Implement slow_mode to return its decay rate and spectral-projection amplitude, including the zero-back-transfer and disconnected limits. The complete scientific reasoning must justify why this mode is the asymptotic physical tail.

For a physical linear population model, a positive eigenvector can identify the dominant decay mode. Handle the zero-back-transfer boundary separately and ensure the measured channel has nonzero tail amplitude.

Returns
-------
np.ndarray, shape (2,), float64: [dominant_decay_rate (s^-1), T1_tail_amplitude for x(0)=(1,0,0)]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def slow_mode(matrix: np.ndarray) -> np.ndarray:
    """Compute the asymptotic T1 contribution of the dominant kinetic mode.

    matrix is a real (3,3) column-vector population generator in s^-1,
    ordered S1,T2,T1. It has nonnegative off-diagonal rates, column sums
    <=0, a stable simple real dominant eigenvalue, and a nonsingular
    eigenvector basis. For x(0)=(1,0,0), return the decay rate and the
    coefficient of its exponential in L(t). The amplitude can be zero
    in a disconnected model: then this mode is not observed in T1.
    Use the spectral projection; do not assume every eigenvalue appears
    with nonzero amplitude. At z=0 the fed lower triplet still has a tail.

    Returns
    -------
    np.ndarray
        Shape (2,), [-dominant_eigenvalue, T1_tail_amplitude].

    Raises
    ------
    ValueError
        If matrix is not finite real shape (3,3), an off-diagonal entry
        is negative, a column sum exceeds 64*eps*max(1,max(abs(matrix))),
        the dominant eigenvalue is nonnegative, nonreal or not simple,
        the eigenvector basis is singular, or results are nonfinite.
        Numerical realness/simplicity uses 1e-10*max(1,abs(dominant)).
        Imaginary amplitude above 1e-10*max(1,abs(real(amplitude))) is invalid.
    """
    return np.empty(2, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_slow_mode(matrix):
    import numpy as np
    try:
        if np.iscomplexobj(matrix):
            raise ValueError("real generator required")
        m = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric generator required") from exc
    if m.shape != (3,3) or not np.all(np.isfinite(m)):
        raise ValueError("finite (3,3) matrix required")
    off = m.copy()
    np.fill_diagonal(off, 0.0)
    tol = 64*np.finfo(float).eps*max(1.0,float(np.max(np.abs(m))))
    if np.any(off < 0) or np.any(m.sum(axis=0) > tol):
        raise ValueError("matrix is not a loss-only population generator")
    try:
        values, vectors = np.linalg.eig(m)
        j = int(np.argmax(values.real))
        pole = values[j]
        etol = 1e-10*max(1.0,abs(pole))
        others = np.delete(values,j)
        if pole.real >= 0 or abs(pole.imag) > etol or np.any(abs(others-pole) <= etol):
            raise ValueError("stable simple real dominant pole required")
        amplitude = vectors[2,j]*np.linalg.inv(vectors)[j,0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("eigendecomposition requires a nonsingular basis") from exc
    if abs(amplitude.imag) > 1e-10*max(1.0,abs(amplitude.real)):
        raise ValueError("nonreal tail amplitude")
    result = np.array([-pole.real, amplitude.real], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite spectral result")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "import numpy as np\nm = np.array([[-1/6e-9-6e7,7.5e5,3.75],[6e7,-750030.,.00015],[0.,30.,-7.75015]])",
            "call": "slow_mode(m)",
            "gold_call": "_oracle_slow_mode(m)"
        },
        {
            "setup": "import numpy as np\nm = np.array([[-1/6e-9-6e7,7.5e5,0.],[6e7,-750030.,0.],[0.,30.,-1/.12]])",
            "call": "slow_mode(m)",
            "gold_call": "_oracle_slow_mode(m)"
        },
        {
            "setup": "import numpy as np\nm = np.diag([-3.,-2.,-1.])",
            "call": "slow_mode(m)",
            "gold_call": "_oracle_slow_mode(m)"
        },
        {
            "setup": "import numpy as np\nm = np.diag([-3.,-2.,0.])\ndef _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(slow_mode, m)",
            "gold_call": "_expect_value_error(_oracle_slow_mode, m)"
        }
    ]
