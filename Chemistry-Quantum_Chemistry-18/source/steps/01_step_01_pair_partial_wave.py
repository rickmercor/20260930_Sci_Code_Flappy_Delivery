"""
Step 01: Legendre coefficients of the correlation factor. Legendre coefficient of order l of the correlation factor p(r12) of a two-electron pair function, as a function of the two radii.

The pair functions of this task have the form Psi(r1_vec, r2_vec) = C p(r12) exp[-omega (r1^2 + r2^2) / 2] with
p(s) = 1 + s / 2 + c s^2, where r12 = |r1_vec - r2_vec|, r1 = |r1_vec|, r2 = |r2_vec| and c >= 0. The linear term fixes the
electron-electron cusp of a singlet. Writing r12 through the angle theta between r1_vec and r2_vec,

  p(r12) = sum over l >= 0 of p_l(r1, r2) P_l(cos theta),

where P_l is the Legendre polynomial of degree l. The coefficients follow from the Legendre expansions of r12 and of
r12^2 = r1^2 + r2^2 - 2 r1 r2 cos theta. The expansion of r12 involves the smaller and larger of the two radii, so each
p_l and its first two derivatives are continuous in (r1, r2) but its third derivative jumps on the line r1 = r2. That
non-analytic behaviour on the diagonal is what makes the natural amplitudes of the pair function decay only algebraically.

Returns
-------
numpy.ndarray with the broadcast shape of r1 and r2: Legendre coefficient p_l(r1, r2) of the correlation factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_partial_wave(r1: "np.ndarray", r2: "np.ndarray", l: int, c: float) -> "np.ndarray":
    '''Legendre coefficient p_l(r1, r2) of p(r12) = 1 + r12/2 + c r12^2.

    Parameters
    ----------
    r1 : np.ndarray
        Radii of electron 1, non-negative; broadcast against r2.
    r2 : np.ndarray
        Radii of electron 2, non-negative; broadcast against r1.
    l : int
        Legendre order, l >= 0.
    c : float
        Coefficient of r12^2 in the correlation factor, c >= 0.

    Returns
    -------
    result : np.ndarray
        Float array with the broadcast shape of r1 and r2 holding p_l(r1, r2). Where both radii are zero only the l = 0
        coefficient is nonzero and equals 1.

    Raises
    ------
    ValueError
        If l is negative.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pair_partial_wave(r1: "np.ndarray", r2: "np.ndarray", l: int, c: float) -> "np.ndarray":
    """Reference implementation."""
    l = int(l)
    if l < 0:
        raise ValueError("the Legendre order must be non-negative")
    a, b = np.broadcast_arrays(np.asarray(r1, dtype=float), np.asarray(r2, dtype=float))
    small = np.minimum(a, b)
    large = np.maximum(a, b)
    safe = np.where(large > 0.0, large, 1.0)
    if l == 0:
        linear = large + small ** 2 / (3.0 * safe)
        value = 1.0 + 0.5 * linear + c * (a ** 2 + b ** 2)
    else:
        ratio = small / safe
        linear = small * ratio ** (l + 1) / (2 * l + 3) - large * ratio ** l / (2 * l - 1)
        value = 0.5 * linear
        if l == 1:
            value = value - 2.0 * c * a * b
    return np.where(large > 0.0, value, 1.0 if l == 0 else 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: s-wave coefficient on a grid crossing the diagonal ---
        {
            "setup": "import numpy as np\n"
                     "r1 = np.array([0.3, 1.0, 2.5, 4.0])[:, None]\n"
                     "r2 = np.array([0.5, 1.0, 3.0])[None, :]\n",
            "call": "pair_partial_wave(r1, r2, 0, 0.05)",
            "gold_call": "_oracle_pair_partial_wave(r1, r2, 0, 0.05)",
            "tol": 1e-12,
        },
        # --- Normal: p-wave, where the r12^2 term contributes ---
        {
            "setup": "import numpy as np\n"
                     "r1 = np.array([0.2, 1.5, 3.0, 6.0])\n"
                     "r2 = np.array([1.0, 1.5, 0.7, 5.0])\n",
            "call": "pair_partial_wave(r1, r2, 1, 0.05)",
            "gold_call": "_oracle_pair_partial_wave(r1, r2, 1, 0.05)",
            "tol": 1e-12,
        },
        # --- Boundary: high order with one radius at the origin and a vanishing r12^2 term ---
        {
            "setup": "import numpy as np\n"
                     "r1 = np.array([0.0, 0.0, 2.0, 3.5])\n"
                     "r2 = np.array([0.0, 1.2, 2.0, 0.4])\n",
            "call": "pair_partial_wave(r1, r2, 4, 0.0)",
            "gold_call": "_oracle_pair_partial_wave(r1, r2, 4, 0.0)",
            "tol": 1e-12,
        },
        # --- Edge: d-wave is untouched by the quadratic term ---
        {
            "setup": "import numpy as np\n"
                     "r1 = np.linspace(0.1, 8.0, 7)[:, None]\n"
                     "r2 = np.linspace(0.05, 9.0, 5)[None, :]\n",
            "call": "pair_partial_wave(r1, r2, 2, 0.3)",
            "gold_call": "_oracle_pair_partial_wave(r1, r2, 2, 0.3)",
            "tol": 1e-12,
        },
        # --- Error: a negative Legendre order must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, 2.0, -1, 0.05)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(pair_partial_wave)",
            "gold_call": "_probe(_oracle_pair_partial_wave)",
        },
    ]
