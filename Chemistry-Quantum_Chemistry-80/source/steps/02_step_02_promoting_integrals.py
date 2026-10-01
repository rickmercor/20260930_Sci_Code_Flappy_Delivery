"""
Franck-Condon overlaps and first-derivative integrals between the vibrational ground state of a mode in the excited electronic state and its ground-state levels, for a Morse bond or, when the anharmonicity is zero, a harmonic mode.

The nonadiabatic coupling that drives internal conversion acts as a first derivative with respect to the promoting coordinate. For a product final state each mode therefore enters either through a Franck-Condon overlap between the vibrational ground state of the initial electronic state S1 and a level of the final state S0, or, for the one mode that promotes the transition, through the integral of the S1 ground state with the derivative of that S0 level. In the model used here the S1 potential of every mode has the same shape as its S0 potential and only its minimum is displaced, so both integrals depend on the anharmonicity and on the displacement.

An anharmonicity of zero denotes a harmonic mode with potential q^2 / 2 in units of its frequency. All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi. Every vibrational eigenfunction is taken real and positive for large positive values of its coordinate (for the Morse functions that is the dissociative side).

Returns
-------
numpy.ndarray of shape (2, n_max + 1): Franck-Condon overlaps g(0, n) in row 0 and derivative integrals t(0, n) in row 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def promoting_integrals(x: float, shift: float, n_max: int) -> np.ndarray:
    '''Overlaps g(0, n) and derivative integrals t(0, n) for one mode.

    Parameters
    ----------
    x : float
        Anharmonicity of the mode; x > 0 means a Morse bond as in
        morse_moment_matrices and x = 0 a harmonic mode.
    shift : float
        Position of the S1 potential minimum on the S0 coordinate (dimensionless);
        the S1 ground state is the S0 ground state translated by shift.
    n_max : int
        Highest S0 level kept; levels 0..n_max are returned.

    Returns
    -------
    integrals : numpy.ndarray
        Real array of shape (2, n_max + 1). Row 0 holds g(0, n) = <chi_0^S1|chi_n^S0>,
        row 1 holds t(0, n) = <chi_0^S1| d/dq chi_n^S0>, the derivative taken with
        respect to the dimensionless coordinate and acting on the S0 level.

    Raises
    ------
    ValueError
        If x is negative, if n_max is not a non-negative integer, or, for x > 0,
        if n_max + 1/2 >= 1 / (2x) - 2.
    '''
    return integrals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def _morse_grid(x, n_max, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    e_top = (n_max + 0.5) - x * (n_max + 0.5) ** 2
    e_top = min(e_top, 0.5 * lam - 1e-3)
    xi_turn = -math.log(1.0 - math.sqrt(2.0 * e_top / lam)) / s
    kappa = math.sqrt(max(lam - 2.0 * e_top, 1e-6))
    lo = -1.6 / s - abs(shift)
    hi = xi_turn + 48.0 / kappa + abs(shift)
    h = 0.004
    xi = np.arange(lo, hi + 0.5 * h, h)
    return xi, h


def _morse_states(x, n_max, xi, shift=0.0):
    s = math.sqrt(2.0 * x)
    lam = 1.0 / (2.0 * x)
    z = 2.0 * lam * np.exp(-s * (xi - shift))
    logz = np.log(2.0 * lam) - s * (xi - shift)
    phi = np.empty((n_max + 1, xi.size))
    dphi = np.empty((n_max + 1, xi.size))
    for n in range(n_max + 1):
        k = lam - n - 0.5
        alpha = 2.0 * lam - 2.0 * n - 1.0
        lognorm = 0.5 * (math.log(s) + math.log(alpha) + gammaln(n + 1.0) - gammaln(2.0 * lam - n))
        base = np.exp(lognorm + k * logz - 0.5 * z)
        lag = eval_genlaguerre(n, alpha, z)
        dlag = -eval_genlaguerre(n - 1, alpha + 1.0, z) if n > 0 else np.zeros_like(z)
        phi[n] = base * lag
        dphi[n] = base * ((k / z - 0.5) * lag + dlag) * (-s * z)
    return phi, dphi


def _oracle_promoting_integrals(x: float, shift: float, n_max: int) -> np.ndarray:
    x = float(x)
    shift = float(shift)
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if int(n_max) != n_max or n_max < 0:
        raise ValueError("n_max must be a non-negative integer")
    n_max = int(n_max)
    if x == 0.0:
        g = np.empty(n_max + 2)
        for n in range(n_max + 2):
            g[n] = math.exp(-shift * shift / 4.0 + n * math.log(abs(shift) / math.sqrt(2.0)) - 0.5 * gammaln(n + 1.0)) if shift != 0.0 else (1.0 if n == 0 else 0.0)
            if shift < 0.0 and n % 2 == 1:
                g[n] = -g[n]
        t = np.empty(n_max + 1)
        for n in range(n_max + 1):
            lower = math.sqrt(n) * g[n - 1] if n > 0 else 0.0
            t[n] = (lower - math.sqrt(n + 1.0) * g[n + 1]) / math.sqrt(2.0)
        return np.array([g[:n_max + 1], t])
    if n_max + 0.5 >= 1.0 / (2.0 * x) - 2.0:
        raise ValueError("n_max must lie well below the dissociation limit")
    xi, h = _morse_grid(x, n_max, shift)
    phi, dphi = _morse_states(x, n_max, xi)
    phi0, _ = _morse_states(x, 0, xi, shift)
    g = phi @ phi0[0] * h
    t = dphi @ phi0[0] * h
    return np.array([g, t])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the N-H bond of the benchmark with its S1 minimum at -0.08, levels 0 to 8 ---
        {
            "setup": """import numpy as np
""",
            "call": "promoting_integrals(0.0215, -0.08, 8)",
            "gold_call": "_oracle_promoting_integrals(0.0215, -0.08, 8)",
            "tol": 1e-09,
        },
        # --- Normal: a Morse bond with a large displacement, levels 0 to 10 ---
        {
            "setup": """import numpy as np
""",
            "call": "promoting_integrals(0.02, 0.6, 10)",
            "gold_call": "_oracle_promoting_integrals(0.02, 0.6, 10)",
            "tol": 1e-09,
        },
        # --- Boundary: no displacement, the overlaps reduce to the ground level alone ---
        {
            "setup": """import numpy as np
""",
            "call": "promoting_integrals(0.0205, 0.0, 6)",
            "gold_call": "_oracle_promoting_integrals(0.0205, 0.0, 6)",
            "tol": 1e-09,
        },
        # --- Edge: a harmonic bath mode (x = 0) displaced by +0.85, levels 0 to 12 ---
        {
            "setup": """import numpy as np
""",
            "call": "promoting_integrals(0.0, 0.85, 12)",
            "gold_call": "_oracle_promoting_integrals(0.0, 0.85, 12)",
            "tol": 1e-10,
        },
        # --- Edge: a harmonic mode displaced in the negative direction ---
        {
            "setup": """import numpy as np
""",
            "call": "promoting_integrals(0.0, -1.3, 6)",
            "gold_call": "_oracle_promoting_integrals(0.0, -1.3, 6)",
            "tol": 1e-10,
        },
    ]
