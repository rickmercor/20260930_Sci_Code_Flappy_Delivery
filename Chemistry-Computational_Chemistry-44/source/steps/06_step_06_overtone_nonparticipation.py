"""
Step 06: Overtone non-participation for a Morse molecule driven at its multiphoton resonance.

Non-participation history of a vibrational overtone for a Morse molecule driven at its multiphoton resonance.

A diatomic molecule is described by its Morse parameters (reduced mass m, well depth D and range alpha, atomic units)
and a polynomial dipole function mu(q) = sum_k c_k q^k. Only its n lowest bound levels v = 0 .. n - 1 are kept, with
their exact Morse eigenvalues E_v and the dipole matrix <v| mu |w> between the exact Morse eigenfunctions. At t = 0 the
molecule is in v = 0 and a linearly polarised continuous-wave laser field E(t) = F cos(omega t) along the bond is
switched on. The laser is tuned to the N-photon resonance of the target overtone v = N, omega = (E_N - E_0) / N, and F
is the peak field of a plane wave whose cycle-averaged intensity in vacuum is I. The ladder then evolves under the full
dipole coupling of the driven-population step, with counter-rotating terms and permanent dipole moments kept.

The populations are recorded every atomic unit of time, t_k = k for k = 0 .. t_end, and the minimal-flow bookkeeping of
the transition-matrix and non-participation steps is applied to this record with the overtone v = N as the target. The
result is the probability P_not(t_k) that the overtone has not yet participated in the flow of probability.

Intensities are given in TW/cm^2 (1 TW/cm^2 = 1e16 W/m^2). Use c = 299792458 m/s and epsilon_0 = 8.8541878128e-12 F/m
for the field in V/m, and 1 atomic unit of electric field = 5.14220674763e11 V/m.

Returns
-------
numpy.ndarray of shape (t_end + 1,), non-participation probability of the overtone at t = 0, 1, ..., t_end atomic time units
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overtone_nonparticipation(intensity: float, t_end: int, n_levels: int, target: int, mass: float, depth: float,
                              alpha: float, dipole_coeffs: "np.ndarray") -> "np.ndarray":
    '''Non-participation probability of the overtone v = target, sampled every atomic time unit up to t_end.

    Parameters
    ----------
    intensity : float
        Cycle-averaged laser intensity in TW/cm^2, positive.
    t_end : int
        Duration of the record in atomic time units, a positive integer.
    n_levels : int
        Number of lowest bound Morse levels kept, at least 2 and larger than target.
    target : int
        Overtone N = 1 .. n_levels - 1; the laser is tuned to omega = (E_N - E_0) / N.
    mass : float
        Reduced mass in electron masses.
    depth : float
        Morse well depth in hartree.
    alpha : float
        Morse range parameter in inverse bohr.
    dipole_coeffs : np.ndarray
        Coefficients [c_0, c_1, ...] of the dipole polynomial in atomic units.

    Returns
    -------
    result : np.ndarray
        Shape (t_end + 1,); element k is P_not at t = k atomic time units, starting from 1.

    Raises
    ------
    ValueError
        If intensity is not positive, t_end is not a positive integer, or target is not between 1 and n_levels - 1,
        and for invalid Morse or dipole inputs as in the dipole-matrix step.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_overtone_nonparticipation(intensity: float, t_end: int, n_levels: int, target: int, mass: float,
                                      depth: float, alpha: float, dipole_coeffs: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not intensity > 0.0:
        raise ValueError("intensity must be positive")
    if int(t_end) != t_end or t_end < 1:
        raise ValueError("t_end must be a positive integer")
    n = int(n_levels)
    order = int(target)
    if not 1 <= order <= n - 1:
        raise ValueError("target must lie between 1 and n_levels - 1")
    dipole = _oracle_morse_dipole_matrix(n, mass, depth, alpha, dipole_coeffs)
    omega0 = alpha * np.sqrt(2.0 * depth / mass)
    v = np.arange(n) + 0.5
    energies = omega0 * v - omega0 ** 2 * v ** 2 / (4.0 * depth)
    field_si = np.sqrt(2.0 * intensity * 1e16 / (299792458.0 * 8.8541878128e-12))
    field = field_si / 5.14220674763e11
    omega = (energies[order] - energies[0]) / order
    populations = _oracle_driven_level_populations(energies, dipole, field, omega, float(t_end), 1.0)
    return _oracle_nonparticipation_probability(populations, order)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: HF, five levels, two-photon drive of v = 2 at 1.2 TW/cm^2 ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "overtone_nonparticipation(1.2, 6000, 5, 2, 1741.312, 0.225019, 1.174145, c)[::200]",
            "gold_call": "_oracle_overtone_nonparticipation(1.2, 6000, 5, 2, 1741.312, 0.225019, 1.174145, c)[::200]",
            "tol": 1e-7,
        },
        # --- Normal: one-photon drive of v = 1 in a two-level truncation ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "overtone_nonparticipation(4.0, 5000, 2, 1, 1741.312, 0.225019, 1.174145, c)[::100]",
            "gold_call": "_oracle_overtone_nonparticipation(4.0, 5000, 2, 1, 1741.312, 0.225019, 1.174145, c)[::100]",
            "tol": 1e-7,
        },
        # --- Boundary: three-photon drive of v = 3 at high intensity in a four-level heavier oscillator ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.43, 0.21, 0.05, -0.012])\n",
            "call": "overtone_nonparticipation(20.0, 4000, 4, 3, 3341.2, 0.1698, 1.0106, c)[::100]",
            "gold_call": "_oracle_overtone_nonparticipation(20.0, 4000, 4, 3, 3341.2, 0.1698, 1.0106, c)[::100]",
            "tol": 1e-7,
        },
        # --- Edge: weak field, the overtone participates only at the 1e-5 level over the record ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "overtone_nonparticipation(0.05, 3000, 3, 2, 1741.312, 0.225019, 1.174145, c)[::100]",
            "gold_call": "_oracle_overtone_nonparticipation(0.05, 3000, 3, 2, 1741.312, 0.225019, 1.174145, c)[::100]",
            "tol": 1e-7,
        },
        # --- Boundary: shallow six-level well driven on its fundamental with three levels kept ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.1, 0.8, -0.05])\n",
            "call": "overtone_nonparticipation(2.0, 4000, 3, 1, 800.0, 0.02, 1.0, c)[::100]",
            "gold_call": "_oracle_overtone_nonparticipation(2.0, 4000, 3, 1, 800.0, 0.02, 1.0, c)[::100]",
            "tol": 1e-7,
        },
        # --- Normal: HF three-photon drive of v = 3 with six levels at 25 TW/cm^2 ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "overtone_nonparticipation(25.0, 5000, 6, 3, 1741.312, 0.225019, 1.174145, c)[::100]",
            "gold_call": "_oracle_overtone_nonparticipation(25.0, 5000, 6, 3, 1741.312, 0.225019, 1.174145, c)[::100]",
            "tol": 1e-7,
        },
        # --- Error: a target equal to the number of kept levels must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, 100, 3, 3, 1741.312, 0.225019, 1.174145, np.array([0.7, 0.3]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(overtone_nonparticipation)",
            "gold_call": "_probe(_oracle_overtone_nonparticipation)",
        },
    ]
