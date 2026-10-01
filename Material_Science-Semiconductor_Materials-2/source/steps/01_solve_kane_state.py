"""
Determine finite-temperature band occupation for the compact MoS2 model.

Represent the conduction band by



\[

\frac{\hbar^2k^2}{2m^*}=E(1+\alpha E),\qquad

D(E)=\frac{g m^*}{2\pi\hbar^2}(1+2\alpha E).

\]



Determine the chemical potential from \(n_s=\int_0^{E_{max}}D(E)f(E,\mu,T)\,dE\), evaluated with Gauss-Legendre quadrature of the supplied order, on the fixed bracket \([-0.5,E_{max}]\) eV, using absolute root tolerance \(10^{-30}\) J and relative tolerance \(10^{-14}\). A density whose chemical potential lies outside this bracket raises `ValueError`. Set \(k_F=k(\max(\mu,0))\).

Returns
-------
A two-entry NumPy array containing the chemical potential in eV and \(k_F\) in m\(^{-1}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_kane_state(temperature_k: float, density_cm2: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Return the chemical potential and Fermi wave vector.

    Parameters
    ----------
    temperature_k : float
        Temperature in kelvin.
    density_cm2 : float
        Electron sheet density in cm^-2.
    mass_ratio : float
        Band-edge effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.
    degeneracy : float
        Combined spin and valley degeneracy.
    energy_max_ev : float
        Upper energy bound in eV.
    quadrature_order : int
        Gauss-Legendre order.

    Returns
    -------
    numpy.ndarray
        Two entries: chemical potential in eV and k_F in m^-1.

    Raises
    ------
    ValueError
        For nonfinite or nonpositive temperature, density, mass ratio,
        degeneracy, or energy maximum; a negative or nonfinite alpha_ev_inv;
        a quadrature order that is not a positive integer (booleans
        included); or a density whose chemical potential lies outside the
        fixed root bracket [-0.5 eV, energy_max_ev] (too dilute a sheet for
        the temperature, or more carriers than the energy interval holds).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Determine finite-temperature band occupation for the compact MoS2 model."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import expit

def _legendre_interval(lower: float, upper: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    scale = 0.5 * (upper - lower)
    return (lower + scale * (nodes + 1.0), scale * weights)
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def _oracle_solve_kane_state(temperature_k: float, density_cm2: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Reference implementation of the finite-temperature density solve."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _K_B = 1.380649e-23
    _M_E = 9.1093837015e-31
    values = (temperature_k, density_cm2, mass_ratio, degeneracy, energy_max_ev)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite thermodynamic and band inputs are required')
    if not np.isfinite(alpha_ev_inv) or alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be finite and nonnegative')
    if isinstance(quadrature_order, (bool, np.bool_)) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order <= 0:
        raise ValueError('quadrature_order must be a positive integer')
    energy_j, weights_j = _legendre_interval(0.0, energy_max_ev * _E_CHARGE, quadrature_order)
    mass = mass_ratio * _M_E
    alpha_j_inv = alpha_ev_inv / _E_CHARGE
    density_of_states = degeneracy * mass / (2.0 * np.pi * _HBAR ** 2) * (1.0 + 2.0 * alpha_j_inv * energy_j)
    target_m2 = density_cm2 * 10000.0

    def _density(chemical_potential_j: float) -> float:
        occupation = expit((chemical_potential_j - energy_j) / (_K_B * temperature_k))
        return float(np.dot(weights_j, density_of_states * occupation))
    lower_j = -0.5 * _E_CHARGE
    upper_j = energy_max_ev * _E_CHARGE
    if _density(upper_j) < target_m2:
        raise ValueError('energy interval does not contain the requested carrier density')
    chemical_potential_j = brentq(lambda value: _density(value) - target_m2, lower_j, upper_j, xtol=1e-30, rtol=1e-14)
    positive_energy = max(chemical_potential_j, 0.0)
    k_fermi = np.sqrt(2.0 * mass * positive_energy * (1.0 + alpha_j_inv * positive_energy)) / _HBAR
    return np.array([chemical_potential_j / _E_CHARGE, k_fermi], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "params = (300.0, 1.2e13, 0.405, 0.7, 4.0, 0.8, 180)",
            "call": "solve_kane_state(*params)",
            "gold_call": "_oracle_solve_kane_state(*params)",
        },
        {
            "setup": "params = (300.0, 2.0e11, 0.405, 0.7, 4.0, 0.8, 160)",
            "call": "solve_kane_state(*params)",
            "gold_call": "_oracle_solve_kane_state(*params)",
        },
        {
            "setup": "params = (80.0, 8.0e12, 0.405, 0.0, 2.0, 0.6, 160)",
            "call": "solve_kane_state(*params)",
            "gold_call": "_oracle_solve_kane_state(*params)",
        },
        {
            "setup": "def bad_argument_sets():\n    good = (300.0, 1.2e13, 0.405, 0.7, 4.0, 0.8, 40)\n    bad = [list(good) for _ in range(17)]\n    bad[0][0] = np.nan\n    bad[1][1] = -1.0e12\n    bad[2][3] = -0.1\n    bad[3][6] = 0\n    bad[4][6] = True\n    bad[5][1] = 1.0e16\n    bad[6][3] = np.nan\n    bad[7][6] = 40.5\n    bad[8][0] = 1200.0\n    bad[8][1] = 2.0e11\n    bad[9][1] = np.inf\n    bad[10][2] = np.nan\n    bad[11][4] = np.inf\n    bad[12][5] = np.nan\n    bad[13][0] = 0.0\n    bad[14][2] = -0.405\n    bad[15][4] = 0.0\n    bad[16][5] = -0.8\n    return bad\ndef rejected_public():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            solve_kane_state(*args)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            _oracle_solve_kane_state(*args)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
