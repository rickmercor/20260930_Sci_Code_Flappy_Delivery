"""
Enumeration of the complete zero-order product states of the local Morse bonds and the harmonic bath whose vibrational excitation energy lies within a resonance window around the electronic energy gap.

At zero temperature the molecule starts in the vibrational ground state of S1, so internal conversion deposits the 0-0 electronic energy gap as vibrational excitation of S0. The zero-order final states are products of local Morse levels, one per bond, and harmonic bath levels, one per mode. Their energies are measured from the zero-point level of each oscillator: a Morse bond contributes its exact bound-level excitation energy and a bath mode contributes its frequency times its quantum number. Only product states whose excitation energy lies within a finite window of the gap take part in the rate.

Each bond keeps levels 0 to n_max, the bath levels are unrestricted apart from the energy bound, and the window is closed (a state exactly on its edge counts). For a reproducible order the states are sorted by energy rounded to 1e-6 cm^-1, and states with equal rounded energy by their quantum numbers in lexicographic order. All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi.

Returns
-------
numpy.ndarray of shape (K, N_x + N_b + 1): quantum numbers and zero-order excitation energy of every product state inside the window
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools

import numpy as np


def window_states(omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, e_gap: float, half_width: float, n_max: int) -> np.ndarray:
    '''Product states within half_width of the energy gap.

    Parameters
    ----------
    omega_loc : numpy.ndarray
        Harmonic frequencies of the N_x local bonds in cm^-1, positive.
    x_loc : numpy.ndarray
        Their anharmonicities, positive, same length as omega_loc.
    omega_bath : numpy.ndarray
        Frequencies of the N_b harmonic bath modes in cm^-1, positive.
    e_gap : float
        0-0 electronic energy gap E_if in cm^-1.
    half_width : float
        Half width of the resonance window in cm^-1, non-negative.
    n_max : int
        Highest Morse level kept for every bond.

    Returns
    -------
    states : numpy.ndarray
        Shape (K, N_x + N_b + 1). Each row holds the local quantum numbers (order of
        omega_loc), the bath quantum numbers (order of omega_bath) and the
        zero-order excitation energy E_n in cm^-1, for every state with
        |e_gap - E_n| <= half_width, sorted as described. An empty window gives
        shape (0, N_x + N_b + 1).

    Raises
    ------
    ValueError
        If omega_loc and x_loc differ in length, a frequency or anharmonicity is
        not positive, half_width is negative, or n_max is not a non-negative
        integer.
    '''
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools

import numpy as np


def _morse_ladder(omega, x, n):
    n = np.asarray(n, dtype=float)
    return omega * (n - x * (n * n + n))


def _oracle_window_states(omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, e_gap: float, half_width: float, n_max: int) -> np.ndarray:
    om_l = np.array(omega_loc, dtype=float).ravel()
    x_l = np.array(x_loc, dtype=float).ravel()
    om_b = np.array(omega_bath, dtype=float).ravel()
    if om_l.size != x_l.size:
        raise ValueError("omega_loc and x_loc must have the same length")
    if np.any(om_l <= 0.0) or np.any(om_b <= 0.0) or np.any(x_l <= 0.0):
        raise ValueError("frequencies and anharmonicities must be positive")
    if half_width < 0.0 or int(n_max) != n_max or n_max < 0:
        raise ValueError("half_width must be non-negative and n_max a non-negative integer")
    n_max = int(n_max)
    top = e_gap + half_width
    levels = [_morse_ladder(om_l[a], x_l[a], np.arange(n_max + 1)) for a in range(om_l.size)]
    found = []

    def _fill_bath(b, acc, e, loc):
        if b == om_b.size:
            if abs(e_gap - e) <= half_width:
                found.append(loc + acc + [e])
            return
        k = 0
        while e + om_b[b] * k <= top:
            _fill_bath(b + 1, acc + [k], e + om_b[b] * k, loc)
            k += 1

    for nl in itertools.product(range(n_max + 1), repeat=om_l.size):
        e = float(sum(levels[a][nl[a]] for a in range(om_l.size)))
        if e > top:
            continue
        _fill_bath(0, [], e, list(nl))
    if not found:
        return np.zeros((0, om_l.size + om_b.size + 1))
    found.sort(key=lambda r: (round(r[-1], 6), r[:-1]))
    return np.array(found, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the benchmark model at 12800 cm^-1 ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3420.0, 3660.0])
x_loc = np.array([0.0205, 0.0215, 0.0225])
delta_loc = np.array([0.12, -0.08, 0.05])
C = np.array([[0.93, 0.25, 0.12], [-0.21, 0.95, 0.20], [0.10, -0.24, 0.97]])
p_xh = np.array([20.0, -12.0, 8.0])
omega_bath = np.array([1605.0, 1460.0, 1180.0, 760.0])
delta_bath = np.array([0.62, 0.35, 0.48, 0.85])
p_bath = np.array([45.0, 20.0, -15.0, 10.0])
cubic = np.array([[0, 0, 1, -180.0], [0, 1, 2, 95.0], [1, 1, 2, -120.0], [0, 2, 2, 60.0], [0, 4, 4, 195.0], [1, 4, 4, -140.0],
                  [2, 3, 4, 105.0], [0, 3, 5, 68.0], [1, 5, 6, 45.0], [2, 6, 6, -38.0], [0, 5, 5, 60.0], [3, 4, 6, 35.0],
                  [4, 5, 6, 18.0], [5, 6, 6, -22.0], [3, 6, 6, 5.5], [4, 6, 6, 5.5], [3, 4, 5, 2.5], [5, 5, 6, 5.0]])
quartic = np.array([[0, 0, 1, 1, 40.0], [0, 1, 2, 2, -25.0], [0, 0, 4, 4, 27.0], [1, 1, 3, 3, -18.0], [0, 1, 4, 6, 14.0],
                    [2, 2, 5, 5, 10.0], [3, 3, 4, 4, 36.0], [4, 4, 4, 4, 60.0], [5, 5, 6, 6, 48.0], [3, 4, 5, 6, 14.0], [4, 4, 4, 6, 66.0]])
thresholds = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 10.0])
""",
            "call": "window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 12800.0, 200.0, 12)",
            "gold_call": "_oracle_window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 12800.0, 200.0, 12)",
            "tol": 1e-09,
        },
        # --- Normal: two bonds and two bath modes at 7000 cm^-1 ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
""",
            "call": "window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 7000.0, 200.0, 8)",
            "gold_call": "_oracle_window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 7000.0, 200.0, 8)",
            "tol": 1e-09,
        },
        # --- Boundary: zero half width with the gap exactly on a product-state energy ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
""",
            "call": "window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 6100.0, 0.0, 8)",
            "gold_call": "_oracle_window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 6100.0, 0.0, 8)",
            "tol": 1e-09,
        },
        # --- Edge: commensurate bath frequencies, so several states share one energy and the lexicographic order decides ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3000.0])
x_loc = np.array([0.02])
omega_bath = np.array([1000.0, 500.0, 250.0])
""",
            "call": "window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 3000.0, 120.0, 4)",
            "gold_call": "_oracle_window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 3000.0, 120.0, 4)",
            "tol": 1e-09,
        },
        # --- Edge: a window below the lowest excitation, which is empty ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3600.0])
x_loc = np.array([0.0205, 0.0220])
delta_loc = np.array([0.15, -0.10])
C = np.array([[0.90, 0.30], [-0.25, 0.95]])
p_xh = np.array([300.0, -180.0])
omega_bath = np.array([1500.0, 800.0])
delta_bath = np.array([0.70, 1.00])
p_bath = np.array([400.0, 150.0])
cubic = np.array([[0, 0, 1, -150.0], [0, 1, 1, 80.0], [1, 1, 1, 90.0], [0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
""",
            "call": "window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 300.0, 100.0, 8)",
            "gold_call": "_oracle_window_states(omega_loc.copy(), x_loc.copy(), omega_bath.copy(), 300.0, 100.0, 8)",
            "tol": 1e-09,
        },
    ]
