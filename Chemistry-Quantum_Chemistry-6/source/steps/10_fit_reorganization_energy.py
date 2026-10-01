"""
Find the reorganisation energy at which the model reproduces a measured rate constant. This is the end-to-end calculation: it starts from the proton potentials of every electronic state and returns one number in eV.

The reorganisation energy is the quantity a rate measurement cannot hand over directly, because it enters the rate in several places at once. It sets the displacement of the product well on the bath coordinate, q_II = (2 * lambda / k) ** 0.5, and with it the crossing point of every reactant-product pair; through those crossing points it sets the tilt of each unpopulated state's proton potential, and so the levels, the proton vibrational functions and the energy denominators that the pathways carry; and it shifts the centre of every spectral overlap. The rate is therefore not a simple function of lambda, and the value that reproduces a measured rate has to be found by solving for it.

Compute once the quantities that do not depend on lambda: the proton grid of n_points uniformly spaced positions from r_min to r_max inclusive, and the direct electronic coupling from the transition dipoles, the displacement vector and the exchange parameters for the given spin case. Then solve

log10 k(lambda) = log10 k_observed

for lambda inside the window [lam_low, lam_high], where k(lambda) is the rate constant of the whole model at that reorganisation energy. The computed rate at the two ends of the window must lie on opposite sides of the observed rate, otherwise no solution is bracketed and that is an error; the window is expected to contain exactly one solution. Return lambda to an absolute accuracy of 1e-10 eV or better.

Returns
-------
float, the reorganisation energy in eV that reproduces the observed rate constant
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_reorganization_energy(k_observed: float, lam_low: float, lam_high: float,
                              r_min: float, r_max: float, n_points: int, r_ref: float,
                              potentials: dict, tilts: dict, virtual_states: tuple,
                              electronic_couplings: dict, d_donor: np.ndarray,
                              d_acceptor: np.ndarray, r_vector: np.ndarray,
                              dipole_unit_factor: float, k0: float, beta: float,
                              spin_case: str, force_constant: float, bath_minima: dict,
                              temperature: float, width: float, n_states: int,
                              hbar2_over_2m: float, hbar: float, kb: float) -> float:
    '''Reorganisation energy that reproduces an observed rate constant.

    Parameters
    ----------
    k_observed : float
        Positive observed rate constant in inverse seconds.
    lam_low, lam_high : float
        Window for the reorganisation energy in eV, 0 < lam_low < lam_high.
    r_min, r_max : float
        Ends of the proton grid in angstrom, r_min < r_max.
    n_points : int
        Odd number of grid points, at least 3.
    r_ref : float
        Positive reduced-coordinate scale of every proton potential in angstrom.
    potentials : dict
        Maps "I", "II" and every label in virtual_states to a tuple (A, B, C)
        in eV of that state's potential at the bottom of its bath well.
    tilts : dict
        Maps every label in virtual_states to the tilt in eV by which the bath
        changes that state's bias.
    virtual_states : tuple of str
        Labels of the unpopulated electronic states.
    electronic_couplings : dict
        Maps a pair of state labels to their electronic coupling in eV; one of
        (a, b) and (b, a) is enough.
    d_donor, d_acceptor : np.ndarray
        (3,) transition dipoles in debye.
    r_vector : np.ndarray
        (3,) donor to acceptor displacement in angstrom.
    dipole_unit_factor : float
        Energy in eV of one debye squared per angstrom cubed.
    k0 : float
        Exchange prefactor in eV.
    beta : float
        Exchange decay constant in inverse angstrom.
    spin_case : str
        "singlet" or "triplet".
    force_constant : float
        Positive bath force constant k in eV.
    bath_minima : dict
        Maps every label in virtual_states to its bath minimum.
    temperature : float
        Positive temperature in kelvin.
    width : float
        Positive standard deviation of each line shape in eV.
    n_states : int
        Number of vibrational levels retained on every state.
    hbar2_over_2m : float
        Positive hbar squared over twice the proton mass in eV angstrom squared.
    hbar : float
        Positive reduced Planck constant in eV seconds.
    kb : float
        Positive Boltzmann constant in eV per kelvin.

    Returns
    -------
    reorganization : float
        The reorganisation energy in eV at which the computed rate constant
        equals k_observed.

    Raises
    ------
    ValueError
        If k_observed is not positive and finite, if the window is not
        0 < lam_low < lam_high, if any other input is missing or invalid, if
        the model is invalid anywhere it is evaluated, or if the computed rate
        at the two ends of the window does not bracket k_observed.
    '''
    return reorganization  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fit_reorganization_energy(k_observed: float, lam_low: float, lam_high: float,
                                      r_min: float, r_max: float, n_points: int,
                                      r_ref: float, potentials: dict, tilts: dict,
                                      virtual_states: tuple, electronic_couplings: dict,
                                      d_donor: np.ndarray, d_acceptor: np.ndarray,
                                      r_vector: np.ndarray, dipole_unit_factor: float,
                                      k0: float, beta: float, spin_case: str,
                                      force_constant: float, bath_minima: dict,
                                      temperature: float, width: float, n_states: int,
                                      hbar2_over_2m: float, hbar: float, kb: float) -> float:
    """Reference implementation."""
    k_obs = float(k_observed)
    if not (np.isfinite(k_obs) and k_obs > 0.0):
        raise ValueError("k_observed must be positive and finite")
    lo, hi = float(lam_low), float(lam_high)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("the window must satisfy 0 < lam_low < lam_high")
    direct = _oracle_direct_coupling(d_donor, d_acceptor, r_vector, dipole_unit_factor,
                                     k0, beta, spin_case)
    target = float(np.log10(k_obs))

    def _mismatch(lam):
        return _oracle_pcent_log_rate(lam, r_min, r_max, n_points, r_ref, potentials, tilts,
                                      virtual_states, electronic_couplings, direct,
                                      force_constant, bath_minima, temperature, width,
                                      n_states, hbar2_over_2m, hbar, kb) - target

    f_lo, f_hi = _mismatch(lo), _mismatch(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError("the observed rate is not bracketed by the window")
    side = 0
    for _ in range(200):
        if hi - lo <= 1e-13:
            break
        if f_hi != f_lo:                        # regula falsi with the Illinois correction
            mid = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
        else:
            mid = 0.5 * (lo + hi)
        if not (lo + 1e-14 < mid < hi - 1e-14):
            mid = 0.5 * (lo + hi)
        f_mid = _mismatch(mid)
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
            if side == 1:
                f_hi *= 0.5
            side = 1
        else:
            hi, f_hi = mid, f_mid
            if side == -1:
                f_lo *= 0.5
            side = -1
        if abs(f_mid) < 1e-14:
            break
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: the five-state assembly with a slower observed rate, which calls for a
# larger reorganisation energy
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "CT2": (0.300, -0.185, 4.260),
       "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.15, "CT2": 0.10, "B": 0.05}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "CT2", "B"), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "singlet", 0.8, qmin, 300.0, 0.185, 3, 2.07500e-3,
        6.582119569e-16, 8.617333262e-5)
""",
            "call": "fit_reorganization_energy(1.0e5, 0.10, 0.60, *args)",
            "gold_call": "_oracle_fit_reorganization_energy(1.0e5, 0.10, 0.60, *args)",
        },
        {
            "setup": """import numpy as np
# normal: a stiffer bath with the unpopulated wells rearranged and no tilt on
# the bridge state
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "CT2": (0.300, -0.185, 4.260),
       "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.10, "CT2": 0.10, "B": 0.0}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
qmin = {"CT1": 0.9, "CT2": 1.2, "B": 0.4}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "CT2", "B"), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "singlet", 2.0, qmin, 300.0, 0.185, 3, 2.07500e-3,
        6.582119569e-16, 8.617333262e-5)
""",
            "call": "fit_reorganization_energy(3.0e5, 0.15, 0.60, *args)",
            "gold_call": "_oracle_fit_reorganization_energy(3.0e5, 0.15, 0.60, *args)",
        },
        {
            "setup": """import numpy as np
# boundary: a coarse grid, two levels per state and a single unpopulated state
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
tilt = {"CT1": 0.12}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
args = (-0.90, 0.90, 81, 0.400, pot, tilt, ("CT1",), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "singlet", 1.0, {"CT1": 1.0}, 300.0, 0.185, 2,
        2.07500e-3, 6.582119569e-16, 8.617333262e-5)
""",
            "call": "fit_reorganization_energy(1.0e6, 0.12, 0.60, *args)",
            "gold_call": "_oracle_fit_reorganization_energy(1.0e6, 0.12, 0.60, *args)",
        },
        {
            "setup": """import numpy as np
# edge: triplet excitations, where the direct part is the exchange term alone
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "CT2": (0.300, -0.185, 4.260),
       "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.15, "CT2": 0.10, "B": 0.05}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "CT2", "B"), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "triplet", 0.8, qmin, 300.0, 0.185, 3, 2.07500e-3,
        6.582119569e-16, 8.617333262e-5)
""",
            "call": "fit_reorganization_energy(1.0e7, 0.10, 0.60, *args)",
            "gold_call": "_oracle_fit_reorganization_energy(1.0e7, 0.10, 0.60, *args)",
        },
        {
            "setup": """import numpy as np
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
tilt = {"CT1": 0.12}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
args = (-0.90, 0.90, 81, 0.400, pot, tilt, ("CT1",), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "singlet", 1.0, {"CT1": 1.0}, 300.0, 0.185, 2,
        2.07500e-3, 6.582119569e-16, 8.617333262e-5)
def run(f):
    try:
        f(1.0e12, 0.12, 0.60, *args); return 0      # a rate the window cannot reach
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(fit_reorganization_energy)",
            "gold_call": "run(_oracle_fit_reorganization_energy)",
        },
        {
            "setup": """import numpy as np
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
tilt = {"CT1": 0.12}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
dd = np.array([1.85, 0.60, -0.40]); da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
args = (-0.90, 0.90, 81, 0.400, pot, tilt, ("CT1",), vel, dd, da, rv,
        0.624151, 17.55, 1.495, "singlet", 1.0, {"CT1": 1.0}, 300.0, 0.185, 2,
        2.07500e-3, 6.582119569e-16, 8.617333262e-5)
def run(f):
    try:
        f(1.0e6, 0.40, 0.20, *args); return 0       # window given the wrong way round
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(fit_reorganization_energy)",
            "gold_call": "run(_oracle_fit_reorganization_energy)",
        },
    ]
