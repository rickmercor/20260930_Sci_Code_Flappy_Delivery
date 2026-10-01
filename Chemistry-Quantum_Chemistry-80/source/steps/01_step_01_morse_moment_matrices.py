"""
Matrix elements of the first four powers of the dimensionless bond coordinate between the lowest bound states of a local X-H Morse oscillator.

In the doorway treatment every physical X-H bond is a one-dimensional Morse oscillator in its own dimensionless stretching coordinate. The cubic and quartic parts of the vibrational force field couple these bonds to each other and to the skeletal bath through products of coordinates, so the anharmonic matrix elements that the later steps need reduce to the one-dimensional integrals <m|xi^p|n> between Morse eigenstates for p = 1 to 4. They have to be evaluated exactly, not in the harmonic approximation, because internal conversion populates high overtones where the Morse and harmonic functions differ strongly.

The Morse potential depends on xi only through the anharmonicity x, so the integrals depend on x and not on the bond frequency. All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi. Every vibrational eigenfunction is taken real and positive for large positive values of its coordinate (for the Morse functions that is the dissociative side).

Returns
-------
numpy.ndarray of shape (4, n_max + 1, n_max + 1) holding <m|xi^p|n> for p = 1..4 between the lowest Morse eigenstates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def morse_moment_matrices(x: float, n_max: int) -> np.ndarray:
    '''Moment matrices <m|xi^p|n> of a Morse oscillator, p = 1..4.

    Parameters
    ----------
    x : float
        Anharmonicity of the bond (dimensionless, positive). The potential in units
        of the harmonic frequency is (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2.
    n_max : int
        Highest vibrational level kept; levels 0..n_max are used.

    Returns
    -------
    moments : numpy.ndarray
        Real array of shape (4, n_max + 1, n_max + 1); moments[p - 1, m, n] is
        <m|xi^p|n> for the bound eigenfunctions m and n, each real, normalised
        and positive for large positive xi.

    Raises
    ------
    ValueError
        If x is not positive, if n_max is not a non-negative integer, or if
        n_max + 1/2 >= 1 / (2x) - 2 (the highest kept level must stay at least two
        levels below the dissociation limit).
    '''
    return moments

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


def _oracle_morse_moment_matrices(x: float, n_max: int) -> np.ndarray:
    x = float(x)
    if not x > 0.0:
        raise ValueError("x must be positive")
    if int(n_max) != n_max or n_max < 0:
        raise ValueError("n_max must be a non-negative integer")
    n_max = int(n_max)
    if n_max + 0.5 >= 1.0 / (2.0 * x) - 2.0:
        raise ValueError("n_max must lie well below the dissociation limit")
    xi, h = _morse_grid(x, n_max)
    phi, _ = _morse_states(x, n_max, xi)
    out = np.empty((4, n_max + 1, n_max + 1))
    for p in range(1, 5):
        out[p - 1] = (phi * xi ** p) @ phi.T * h
    return 0.5 * (out + np.transpose(out, (0, 2, 1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the C-H bond of the benchmark, levels 0 to 6 ---
        {
            "setup": """import numpy as np
""",
            "call": "morse_moment_matrices(0.0205, 6)",
            "gold_call": "_oracle_morse_moment_matrices(0.0205, 6)",
            "tol": 1e-07,
        },
        # --- Normal: a strongly anharmonic bond, levels 0 to 8 ---
        {
            "setup": """import numpy as np
""",
            "call": "morse_moment_matrices(0.03, 8)",
            "gold_call": "_oracle_morse_moment_matrices(0.03, 8)",
            "tol": 1e-07,
        },
        # --- Boundary: only the ground level ---
        {
            "setup": """import numpy as np
""",
            "call": "morse_moment_matrices(0.0225, 0)",
            "gold_call": "_oracle_morse_moment_matrices(0.0225, 0)",
            "tol": 1e-08,
        },
        # --- Edge: a weakly anharmonic bond, where the moments approach the harmonic-oscillator values ---
        {
            "setup": """import numpy as np
""",
            "call": "morse_moment_matrices(0.006, 5)",
            "gold_call": "_oracle_morse_moment_matrices(0.006, 5)",
            "tol": 1e-06,
        },
    ]
