"""
With the system assembled the scattering amplitudes follow from one dense solve. The unknown vector holds, in order, the forward interior amplitudes, the backward interior amplitudes, the reflected amplitudes and the transmitted amplitudes, each block running over its index from minus J to plus J, so the two exterior blocks are recovered by slicing the solution rather than by any further algebra. The scattering coefficients quoted for a unit incident amplitude are the magnitudes of those complex amplitudes.

The conditioning of the system deserves a look rather than a trust. The interior modes accumulate phase across the segment, and where the segment sits inside a wavenumber bandgap the two nearly degenerate modes that dominate the response carry almost the same phase, which brings their columns close to parallel. The condition number is returned so that a configuration in which the solve has lost its meaning can be recognised; in the regime this task prescribes it stays modest, and a value orders of magnitude larger is the signal that the segment has been pushed into a range where the steady scattering problem no longer has a well conditioned answer.

A second and more physical caution belongs here. The linear system returns a number for any parameters it is handed, including parameters for which the bounded segment has no steady state at all, because a sufficiently deep or sufficiently long modulation drives the system parametrically unstable and the response then grows without bound in time. Nothing in this solve detects that: the algebra is the algebra of a steady response that has been assumed to exist. Whether the prescribed configuration admits such a steady state is not settled by anything computed here; it is assumed, and the coefficients returned are conditional on that assumption.

Returns
-------
dict holding a float64 transmission of shape (2 J + 1,) and a float64 reflection of the same shape, each indexed by scattering order from minus J to plus J and giving the magnitude of that order relative to a unit incident amplitude; and a native float condition_number for the system matrix in the two-norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_scattering_coefficients(matrix: np.ndarray, rhs: np.ndarray, j_order: int) -> dict:
    """Solve the interface system and read off the transmission and reflection magnitudes at every retained order.

    Parameters
    ----------
    matrix : np.ndarray
        The interface system matrix, shape (8 J + 4, 8 J + 4).
    rhs : np.ndarray
        Its right-hand side for unit incident amplitude, shape (8 J + 4,).
    j_order : int
        The mode-coupling order J, zero or more.

    Returns
    -------
    dict
        Under the keys transmission, reflection and condition_number. The
        transmission and reflection entries are float64 arrays of shape (2 J + 1,)
        indexed by scattering order from minus J to plus J. The condition_number
        entry is a native float.

    Raises
    ------
    ValueError
        If matrix is not square, if its size is not four times two J plus one, if
        rhs does not match it, if either is not finite, if j_order is negative, or
        if the system is singular.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_harmonic_scattering_coefficients(matrix: np.ndarray, rhs: np.ndarray, j_order: int) -> dict:
    matrix = np.asarray(matrix, dtype=np.complex128)
    rhs = np.asarray(rhs, dtype=np.complex128)
    if int(j_order) != j_order or int(j_order) < 0:
        raise ValueError("j_order must be a non-negative integer")
    j_order = int(j_order)
    width = 2 * j_order + 1
    size = 4 * width
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("matrix must be square of side four times two j_order plus one")
    if rhs.ndim != 1 or rhs.size != size:
        raise ValueError("rhs must match the side of matrix")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("matrix and rhs must be finite")

    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise ValueError("the interface system is singular")
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the interface system is singular") from exc

    reflection = np.abs(solution[2 * width:3 * width]).astype(np.float64)
    transmission = np.abs(solution[3 * width:4 * width]).astype(np.float64)
    return {
        "transmission": transmission,
        "reflection": reflection,
        "condition_number": condition_number,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng = np.random.default_rng(3)\nj = 2\nn = 4 * (2 * j + 1)\nM = np.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)), dtype=complex)\nB = np.asarray(rng.normal(size=n) + 1j * rng.normal(size=n), dtype=complex)\ndef pack(d):\n    transmission = np.asarray(d['transmission'])\n    reflection = np.asarray(d['reflection'])\n    shape = np.asarray(transmission.shape + reflection.shape, dtype=float)\n    return np.concatenate((shape, transmission.ravel(), reflection.ravel(), np.asarray([d['condition_number']], dtype=float)))\n",
            "call": 'pack(harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
            "gold_call": 'pack(_oracle_harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
        },
        {
            "setup": "import numpy as np\nj = 0\nM = np.asarray(np.eye(4) * 2.0, dtype=complex)\nB = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=complex)\ndef pack(d):\n    transmission = np.asarray(d['transmission'])\n    reflection = np.asarray(d['reflection'])\n    shape = np.asarray(transmission.shape + reflection.shape, dtype=float)\n    return np.concatenate((shape, transmission.ravel(), reflection.ravel(), np.asarray([d['condition_number']], dtype=float)))\n",
            "call": 'pack(harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
            "gold_call": 'pack(_oracle_harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
        },
        {
            "setup": "import numpy as np\nj = 1\nn = 12\nM = np.asarray(np.diag(np.arange(1.0, n + 1.0)), dtype=complex)\nB = np.asarray(np.ones(n), dtype=complex)\ndef pack(d):\n    transmission = np.asarray(d['transmission'])\n    reflection = np.asarray(d['reflection'])\n    shape = np.asarray(transmission.shape + reflection.shape, dtype=float)\n    return np.concatenate((shape, transmission.ravel(), reflection.ravel(), np.asarray([d['condition_number']], dtype=float)))\n",
            "call": 'pack(harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
            "gold_call": 'pack(_oracle_harmonic_scattering_coefficients(np.array(M, copy=True),np.array(B, copy=True), j))',
        },
        {
            "setup": "import numpy as np\nj = 1\nM = np.asarray(np.eye(12), dtype=complex)\nB = np.asarray(np.ones(12), dtype=complex)\nRECT = np.asarray(np.ones((12, 11)), dtype=complex)\nSING = np.asarray(np.zeros((12, 12)), dtype=complex)\nNAN = np.asarray(np.eye(12) * float('nan'), dtype=complex)\nSHORT = np.asarray(np.ones(11), dtype=complex)\ndef verdict(fn, m=M, b=B, jj=1):\n    try:\n        fn(np.array(m, copy=True), np.array(b, copy=True), jj)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": '(verdict(harmonic_scattering_coefficients, m=RECT), verdict(harmonic_scattering_coefficients, m=SING), verdict(harmonic_scattering_coefficients, m=NAN), verdict(harmonic_scattering_coefficients, b=SHORT), verdict(harmonic_scattering_coefficients, jj=-1), verdict(harmonic_scattering_coefficients, jj=2), verdict(harmonic_scattering_coefficients))',
            "gold_call": '(verdict(_oracle_harmonic_scattering_coefficients, m=RECT), verdict(_oracle_harmonic_scattering_coefficients, m=SING), verdict(_oracle_harmonic_scattering_coefficients, m=NAN), verdict(_oracle_harmonic_scattering_coefficients, b=SHORT), verdict(_oracle_harmonic_scattering_coefficients, jj=-1), verdict(_oracle_harmonic_scattering_coefficients, jj=2), verdict(_oracle_harmonic_scattering_coefficients))',
        },
    ]
