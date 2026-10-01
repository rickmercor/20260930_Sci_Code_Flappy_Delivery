"""
Assemble the matrix of the growth-direction momentum in the hard-wall envelope basis.

Confinement is modelled by a hard-wall well of width L centred on the origin, whose normalised envelopes are phi_n(z) = sqrt(2/L) cos(n pi z / L) for odd n and sqrt(2/L) sin(n pi z / L) for even n, with n = 1, 2, ... ordered by increasing n. Fix that phase convention exactly, since the individual matrix elements depend on it. The squared growth momentum is diagonal in this basis but the momentum operator itself is not: it connects envelopes of opposite parity, and those off-diagonal elements are what the pair terms of the kinetic energy need. Return the matrix of p_z in units of 1/nm with hbar set to one. It is Hermitian and, in this real basis, purely imaginary.

Returns
-------
numpy.ndarray of shape (num_modes, num_modes), complex, in 1/nm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def momentum_matrix_elements(width: float, num_modes: int) -> "np.ndarray":
    '''Matrix of the growth-direction momentum in the hard-wall envelope basis.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    num_modes : int
        Number of envelopes retained, n = 1 .. num_modes. Must be an integer >= 1.

    Returns
    -------
    numpy.ndarray
        Hermitian complex array of shape (num_modes, num_modes) in 1/nm.

    Raises
    ------
    ValueError
        If width is not finite or not positive, or num_modes is not an integer >= 1.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_momentum_matrix_elements(width: float, num_modes: int) -> "np.ndarray":
    lw = float(width)
    if not np.isfinite(lw) or lw <= 0.0:
        raise ValueError("width must be finite and strictly positive")
    if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)) or int(num_modes) < 1:
        raise ValueError("num_modes must be an integer >= 1")
    m = int(num_modes)
    d = np.zeros((m, m))
    for n in range(1, m + 1):
        for p in range(1, m + 1):
            if (n + p) % 2 == 0 or n % 2 == 0:
                continue
            d[n - 1, p - 1] = 4.0 * n * p / (lw * (n * n - p * p)) * (-1) ** (((n - p - 1) // 2) % 2)
    return -1j * (d - d.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "momentum_matrix_elements(6.4, 16)",
            "gold_call": "_oracle_momentum_matrix_elements(6.4, 16)",
        },
        {
            "setup": "import numpy as np",
            "call": "momentum_matrix_elements(10.0, 4)",
            "gold_call": "_oracle_momentum_matrix_elements(10.0, 4)",
        },
        {
            "setup": "import numpy as np",
            "call": "momentum_matrix_elements(3.75, 7)",
            "gold_call": "_oracle_momentum_matrix_elements(3.75, 7)",
        },
    ]
