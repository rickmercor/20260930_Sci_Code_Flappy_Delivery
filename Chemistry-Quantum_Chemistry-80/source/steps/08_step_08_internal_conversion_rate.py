"""
State-correlated internal-conversion rate constant of the local-Morse doorway model: every product state in the window contributes its first-order intensity with its own spreading width and its own energy mismatch.

The rate constant is a golden-rule sum over the product final states in the resonance window. Each state contributes its transition intensity I_n times the Lorentzian factor Gamma_n / (Delta_n^2 + Gamma_n^2 / 4), with Gamma_n its full width and Delta_n = E_if - E_n its energy mismatch, and the sum, in cm^-1 because every energy is in cm^-1, becomes a rate in s^-1 on multiplication by 2 pi c with c = 2.99792458e10 cm/s. In the state-correlated form each state uses its own first-order corrected intensity and its own spreading width, correlated state by state; a state whose width is zero contributes nothing.

The pipeline is window_states, local_force_field, first_order_intensities and spreading_widths, with the same model parameters throughout. All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi.

Returns
-------
float: state-correlated internal-conversion rate constant k_IC in s^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np


def internal_conversion_rate(omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, C: np.ndarray, p_xh: np.ndarray, omega_bath: np.ndarray, delta_bath: np.ndarray, p_bath: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, thresholds: np.ndarray, e_gap: float, n_max: int, half_width: float, resonance_cutoff: float, eta: float) -> float:
    '''State-correlated internal-conversion rate constant in s^-1.

    Parameters
    ----------
    omega_loc, x_loc, delta_loc : numpy.ndarray
        Harmonic frequencies (cm^-1), anharmonicities and S1 minimum positions of
        the N_x local bonds.
    C : numpy.ndarray
        Local-bond transformation, shape (N_x, M_x).
    p_xh : numpy.ndarray
        Coupling coefficients of the M_x X-H normal modes, cm^-1.
    omega_bath, delta_bath, p_bath : numpy.ndarray
        Frequencies (cm^-1), S1 minimum positions and coupling coefficients
        (cm^-1) of the N_b bath modes.
    cubic, quartic : numpy.ndarray
        Normal-coordinate force field as in local_force_field.
    thresholds : numpy.ndarray
        Screening thresholds as in local_force_field.
    e_gap : float
        0-0 energy gap E_if in cm^-1.
    n_max : int
        Highest Morse level kept for every bond.
    half_width : float
        Half width of the resonance window and largest relaxation-step mismatch,
        cm^-1.
    resonance_cutoff : float
        Resonance cutoff of the first-order correction, cm^-1.
    eta : float
        Half width of the Lorentzian regulator in the spreading widths, cm^-1.

    Returns
    -------
    rate : float
        The internal-conversion rate constant k_IC in s^-1 (0.0 for an empty
        window).

    Raises
    ------
    ValueError
        If resonance_cutoff or half_width is negative or eta is not positive, and
        under the conditions listed for window_states, local_force_field,
        zero_order_amplitudes, first_order_intensities and spreading_widths.
    '''
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_internal_conversion_rate(omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, C: np.ndarray, p_xh: np.ndarray, omega_bath: np.ndarray, delta_bath: np.ndarray, p_bath: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, thresholds: np.ndarray, e_gap: float, n_max: int, half_width: float, resonance_cutoff: float, eta: float) -> float:
    if resonance_cutoff < 0.0 or half_width < 0.0 or not eta > 0.0:
        raise ValueError("resonance_cutoff and half_width must be non-negative and eta positive")
    n_b = np.array(omega_bath, dtype=float).size
    states = _oracle_window_states(omega_loc, x_loc, omega_bath, e_gap, half_width, n_max)
    if states.shape[0] == 0:
        return 0.0
    terms = _oracle_local_force_field(C, cubic, quartic, n_b, thresholds)
    inten = _oracle_first_order_intensities(states, terms, C, p_xh, p_bath, omega_loc, x_loc, delta_loc, delta_bath, n_max, resonance_cutoff)
    width = _oracle_spreading_widths(states, terms, omega_loc, x_loc, omega_bath, n_max, eta, half_width)
    detune = e_gap - states[:, -1]
    shape = np.zeros(states.shape[0])
    live = width > 0.0
    shape[live] = width[live] / (detune[live] ** 2 + 0.25 * width[live] ** 2)
    return float(2.0 * math.pi * 2.99792458e10 * np.sum(inten * shape))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the two-bond model at 7000 cm^-1 ---
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
            "call": "internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 7000.0, 8, 200.0, 10.0, 20.0)",
            "gold_call": "_oracle_internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 7000.0, 8, 200.0, 10.0, 20.0)",
            "tol": 1e-07,
        },
        # --- Normal: the benchmark model at 9000 cm^-1 with a 150 cm^-1 window ---
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
            "call": "internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 9000.0, 10, 150.0, 10.0, 20.0)",
            "gold_call": "_oracle_internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 9000.0, 10, 150.0, 10.0, 20.0)",
            "tol": 1e-07,
        },
        # --- Boundary: no pure X-H anharmonicity, the rate uses the zero-order intensities ---
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
cubic = np.array([[0, 2, 2, 180.0], [1, 2, 3, 60.0], [0, 3, 3, 40.0],
                  [2, 3, 3, 25.0], [2, 2, 3, -12.0]])
quartic = np.array([[0, 0, 2, 2, 20.0], [2, 2, 3, 3, 44.0], [0, 1, 2, 3, 15.0]])
thresholds = np.array([0.5, 0.5, 2.0, 2.0, 2.0, 5.0])
""",
            "call": "internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 7000.0, 8, 200.0, 10.0, 20.0)",
            "gold_call": "_oracle_internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 7000.0, 8, 200.0, 10.0, 20.0)",
            "tol": 1e-07,
        },
        # --- Edge: a window below the lowest excitation, the rate is zero ---
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
            "call": "internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 300.0, 8, 100.0, 10.0, 20.0)",
            "gold_call": "_oracle_internal_conversion_rate(omega_loc.copy(), x_loc.copy(), delta_loc.copy(), C.copy(), p_xh.copy(), omega_bath.copy(), delta_bath.copy(), p_bath.copy(), cubic.copy(), quartic.copy(), thresholds.copy(), 300.0, 8, 100.0, 10.0, 20.0)",
            "tol": 1e-07,
        },
    ]
