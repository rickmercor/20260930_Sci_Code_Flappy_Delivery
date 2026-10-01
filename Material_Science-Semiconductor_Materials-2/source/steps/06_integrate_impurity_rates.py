"""
Integrate the screened Coulomb rate coefficient over scattering angle.

Integrate the screened potential from step 04 as



\[

\Gamma_I(E;N_I)=\frac{N_I m^*(1+2\alpha E)}{2\pi\hbar^3}

\int_0^{2\pi}|U(q)|^2(1-\cos\phi)\,d\phi.

\]



This elastic rate also uses the initial-state Kane density-of-states factor. The public result is the coefficient per impurity density in cm\(^{-2}\), so include the \(10^4\) conversion to m\(^{-2}\).

Returns
-------
A NumPy vector of impurity rate coefficients in s\(^{-1}\) cm\(^2\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_impurity_rate_coefficient(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', screened_potential_j_m2: 'np.ndarray', mass_ratio: float, alpha_ev_inv: float) -> 'np.ndarray':
    """Return the charged-impurity rate per sheet density in cm^-2.

    The elastic prefactor uses the initial-state Kane density-of-states
    multiplier (1 + 2*alpha*E).

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Carrier energies in eV.
    scattering_angle_rad : numpy.ndarray
        Scattering angles in radians.
    angle_weights : numpy.ndarray
        Quadrature weights over zero to two pi.
    screened_potential_j_m2 : numpy.ndarray
        Screened potential with shape (energy, angle), in J m^2.
    mass_ratio : float
        Effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.

    Returns
    -------
    numpy.ndarray
        Momentum-relaxation-rate coefficients in s^-1 cm^2.

    Raises
    ------
    ValueError
        For non-one-dimensional energy or angle arrays, weights not matching
        the angles, a potential shape other than (n_energy, n_angle),
        nonfinite entries, negative energies, nonpositive weights or
        potential values, a nonpositive or nonfinite mass ratio, or a
        negative or nonfinite alpha_ev_inv.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Integrate the screened Coulomb rate coefficient over scattering angle."""
import numpy as np
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def _oracle_integrate_impurity_rate_coefficient(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', screened_potential_j_m2: 'np.ndarray', mass_ratio: float, alpha_ev_inv: float) -> 'np.ndarray':
    """Reference elastic charged-defect coefficient for density in cm^-2."""
    _HBAR = 1.054571817e-34
    _M_E = 9.1093837015e-31
    energy = np.asarray(energy_ev, dtype=float)
    angle = np.asarray(scattering_angle_rad, dtype=float)
    weights = np.asarray(angle_weights, dtype=float)
    potential = np.asarray(screened_potential_j_m2, dtype=float)
    if energy.ndim != 1 or angle.ndim != 1 or weights.shape != angle.shape:
        raise ValueError('energy, angle, and angle_weights must be one-dimensional')
    if potential.shape != (energy.size, angle.size):
        raise ValueError('screened_potential_j_m2 has an incompatible shape')
    arrays = (energy, angle, weights, potential)
    if not all((np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError('all arrays must contain finite values')
    if np.any(energy < 0.0) or np.any(weights <= 0.0) or np.any(potential <= 0.0):
        raise ValueError('energy must be nonnegative and weights and potential positive')
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError('mass_ratio must be positive and finite')
    if not np.isfinite(alpha_ev_inv) or alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be finite and nonnegative')
    angular_integral = potential ** 2 * (1.0 - np.cos(angle))[None, :] @ weights
    prefactor = 10000.0 * mass_ratio * _M_E * (1.0 + 2.0 * alpha_ev_inv * energy) / (2.0 * np.pi * _HBAR ** 3)
    return prefactor * angular_integral

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "energy = np.array([0.01, 0.05]); angle = np.array([0.2, 1.5, 4.0]); weights = np.array([1.0, 2.0, 3.283185307179586]); potential = np.full((2, 3), 2e-37); params = (energy, angle, weights, potential, 0.405, 0.7)",
            "call": "integrate_impurity_rate_coefficient(*params)",
            "gold_call": "_oracle_integrate_impurity_rate_coefficient(*params)",
        },
        {
            "setup": "energy = np.array([0.0]); angle = np.array([np.pi]); weights = np.array([2*np.pi]); potential = np.array([[1e-39]]); params = (energy, angle, weights, potential, 0.405, 0.0)",
            "call": "integrate_impurity_rate_coefficient(*params)",
            "gold_call": "_oracle_integrate_impurity_rate_coefficient(*params)",
        },
        {
            "setup": "energy = np.linspace(0.0, 0.8, 4); angle = np.linspace(0.1, 6.1, 5); weights = np.full(5, 2*np.pi/5); potential = np.geomspace(1e-40, 1e-36, 20).reshape(4, 5); params = (energy, angle, weights, potential, 1.5, 2.0)",
            "call": "integrate_impurity_rate_coefficient(*params)",
            "gold_call": "_oracle_integrate_impurity_rate_coefficient(*params)",
        },
        {
            "setup": "def bad_argument_sets():\n    good = (np.array([0.01, 0.05]), np.array([1.0, 4.0]), np.array([3.0, 3.283185307179586]), np.full((2, 2), 1.0e-37), 0.405, 0.7)\n    bad = [list(good) for _ in range(12)]\n    bad[0][3] = np.full((2, 3), 1.0e-37)\n    bad[1][3] = np.zeros((2, 2))\n    bad[2][0] = np.array([-0.01, 0.05])\n    bad[3][4] = 0.0\n    bad[4][5] = np.nan\n    bad[5][0] = np.array([[0.01, 0.05]])\n    bad[6][2] = np.array([3.0, 3.0, 0.283185307179586])\n    bad[7][3] = np.array([[1.0e-37, np.nan], [1.0e-37, 1.0e-37]])\n    bad[8][4] = np.inf\n    bad[9][5] = -0.1\n    bad[10][1] = np.array([[1.0, 4.0]])\n    bad[10][2] = np.array([[3.0, 3.283185307179586]])\n    bad[11][2] = np.array([3.0, 0.0])\n    return bad\ndef rejected_public():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            integrate_impurity_rate_coefficient(*args)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            _oracle_integrate_impurity_rate_coefficient(*args)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
