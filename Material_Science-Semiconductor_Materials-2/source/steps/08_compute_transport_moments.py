"""
Convert energy-resolved relaxation rates into thermoelectric and Hall responses.

Combine independent rates by \(\tau^{-1}=\Gamma_{ac}+\Gamma_I+\Gamma_{SO}\), then use



\[

D(E)=\frac{g m^*}{2\pi\hbar^2}(1+2\alpha E),\qquad

v(E)=\frac{\hbar k(E)}{m^*(1+2\alpha E)},

\]



\[

L_j=\frac12\int D(E)v(E)^2\tau(E)\left(-\frac{\partial f}{\partial E}\right)(E-\mu)^j\,dE .

\]



Return \(\sigma=e^2L_0/a\), \(S=-L_1/(eTL_0)\), and \(P=10^3S^2\sigma\); an empty grid, or a grid on which \(L_0=0\), raises `ValueError`. The fourth entry is \(|\sigma_{xy}|/B\): the magnitude of the sheet Hall conductivity per unit perpendicular magnetic field, to first order in \(B\), that the same isotropic energy-resolved relaxation-time Boltzmann equation gives for this band and these rates when the Lorentz force is included.

Returns
-------
A four-entry NumPy array containing conductivity in S m\(^{-1}\), Seebeck coefficient in microV K\(^{-1}\), power factor in mW m\(^{-1}\) K\(^{-2}\), and \(|\sigma_{xy}|/B\) in microS T\(^{-1}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_transport_moments(energy_ev: 'np.ndarray', energy_weights_ev: 'np.ndarray', chemical_potential_ev: float, acoustic_rate_s_inv: 'np.ndarray', impurity_rate_s_inv: 'np.ndarray', surface_optical_rate_s_inv: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, thickness_angstrom: float) -> 'np.ndarray':
    """Return conductivity, Seebeck coefficient, power factor, and Hall response.

    Parameters
    ----------
    energy_ev, energy_weights_ev : numpy.ndarray
        One-dimensional energy nodes and positive quadrature weights in eV.
        Energies must be nonnegative.
    chemical_potential_ev : float
        Chemical potential relative to the band edge in eV.
    acoustic_rate_s_inv, impurity_rate_s_inv, surface_optical_rate_s_inv : numpy.ndarray
        Nonnegative independent momentum-relaxation rates in s^-1 on the
        energy nodes. Their sum must be positive at every node.
    temperature_k, mass_ratio, degeneracy : float
        Positive temperature in kelvin, band-edge mass ratio, and combined
        spin and valley degeneracy.
    alpha_ev_inv : float
        Nonnegative nonparabolicity in eV^-1.
    thickness_angstrom : float
        Positive thickness used to convert sheet conductance to S m^-1.

    Returns
    -------
    numpy.ndarray
        Four entries: conductivity in S m^-1, Seebeck coefficient in
        microvolt K^-1, power factor in mW m^-1 K^-2, and |sigma_xy|/B in
        microsiemens T^-1, the magnitude of the sheet Hall conductivity per
        unit perpendicular magnetic field to first order in B, obtained from
        the same isotropic relaxation-time Boltzmann equation for this band
        and these rates with the Lorentz force included.

    Raises
    ------
    ValueError
        For mismatched or non-one-dimensional arrays, nonfinite values,
        negative energies or rates, nonpositive weights or total rate,
        nonpositive temperature, mass, degeneracy, or thickness, a negative
        alpha_ev_inv, or a grid on which the conductivity moment L_0 is zero
        (an empty grid, every node at the band edge, or a thermal window that
        underflows at every node).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Convert energy-resolved relaxation rates into thermoelectric and Hall responses."""
import numpy as np
from scipy.special import expit
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def _oracle_compute_transport_moments(energy_ev: 'np.ndarray', energy_weights_ev: 'np.ndarray', chemical_potential_ev: float, acoustic_rate_s_inv: 'np.ndarray', impurity_rate_s_inv: 'np.ndarray', surface_optical_rate_s_inv: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, thickness_angstrom: float) -> 'np.ndarray':
    """Reference two-dimensional Boltzmann moment calculation."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _K_B = 1.380649e-23
    _M_E = 9.1093837015e-31
    energy = np.asarray(energy_ev, dtype=float)
    weights_ev = np.asarray(energy_weights_ev, dtype=float)
    acoustic = np.asarray(acoustic_rate_s_inv, dtype=float)
    impurity = np.asarray(impurity_rate_s_inv, dtype=float)
    surface = np.asarray(surface_optical_rate_s_inv, dtype=float)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError('energy_ev must be a nonempty one-dimensional array')
    if not weights_ev.shape == acoustic.shape == impurity.shape == surface.shape == energy.shape:
        raise ValueError('energy-grid arrays must have matching shapes')
    arrays = (energy, weights_ev, acoustic, impurity, surface)
    if not all((np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError('energy-grid arrays must contain finite values')
    if np.any(energy < 0.0) or np.any(weights_ev <= 0.0):
        raise ValueError('energy must be nonnegative and weights positive')
    if any((np.any(rate < 0.0) for rate in (acoustic, impurity, surface))):
        raise ValueError('independent rates must be nonnegative')
    total_rate = acoustic + impurity + surface
    if np.any(total_rate <= 0.0):
        raise ValueError('the total rate must be strictly positive')
    values = np.asarray((temperature_k, mass_ratio, degeneracy, thickness_angstrom, chemical_potential_ev, alpha_ev_inv), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('scalar transport inputs must be finite')
    if min(temperature_k, mass_ratio, degeneracy, thickness_angstrom) <= 0.0:
        raise ValueError('temperature, mass, degeneracy, and thickness must be positive')
    if alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be nonnegative')
    energy_j = energy * _E_CHARGE
    weights_j = weights_ev * _E_CHARGE
    mass = mass_ratio * _M_E
    kane_factor = 1.0 + 2.0 * alpha_ev_inv * energy
    density_of_states = degeneracy * mass / (2.0 * np.pi * _HBAR ** 2) * kane_factor
    wave_vector = np.sqrt(2.0 * mass * energy_j * (1.0 + alpha_ev_inv * energy)) / _HBAR
    cyclotron_mass = mass * kane_factor
    velocity = _HBAR * wave_vector / cyclotron_mass
    occupation = expit((chemical_potential_ev - energy) * _E_CHARGE / (_K_B * temperature_k))
    minus_derivative = occupation * (1.0 - occupation) / (_K_B * temperature_k)
    relaxation_time = 1.0 / total_rate
    transport_density = density_of_states * velocity ** 2 * relaxation_time * minus_derivative / 2.0
    moment_0 = float(np.dot(weights_j, transport_density))
    moment_1 = float(np.dot(weights_j, transport_density * (energy_j - chemical_potential_ev * _E_CHARGE)))
    hall_moment = float(np.dot(weights_j, transport_density * relaxation_time / cyclotron_mass))
    if moment_0 <= 0.0:
        raise ValueError('the conductivity moment is zero on the supplied grid')
    conductivity = _E_CHARGE ** 2 * moment_0 / (thickness_angstrom * 1e-10)
    seebeck_v_k = -moment_1 / (_E_CHARGE * temperature_k * moment_0)
    power_factor_mw = seebeck_v_k ** 2 * conductivity * 1000.0
    hall_conductivity_us_t = _E_CHARGE ** 3 * hall_moment * 1000000.0
    return np.array([conductivity, seebeck_v_k * 1000000.0, power_factor_mw, hall_conductivity_us_t], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "energy = np.array([0.01, 0.04, 0.10]); weights = np.array([0.02, 0.04, 0.06]); ac = np.array([1e13, 1.1e13, 1.2e13]); imp = np.array([2e13, 1.8e13, 1.5e13]); so = np.array([5e12, 8e12, 1.4e13]); params = (energy, weights, 0.025, ac, imp, so, 300.0, 0.405, 0.7, 4.0, 6.6)",
            "call": "compute_transport_moments(*params)",
            "gold_call": "_oracle_compute_transport_moments(*params)",
        },
        {
            "setup": "energy = np.array([0.0, 1e-6, 0.02]); weights = np.array([5e-7, 5e-7, 0.01]); ac = np.array([1e12, 1e12, 3e12]); imp = np.zeros(3); so = np.array([0.0, 0.0, 2e12]); params = (energy, weights, 0.0, ac, imp, so, 120.0, 0.405, 0.0, 2.0, 6.6)",
            "call": "compute_transport_moments(*params)",
            "gold_call": "_oracle_compute_transport_moments(*params)",
        },
        {
            "setup": "energy = np.geomspace(1e-6, 0.8, 12); weights = np.full(12, 0.8/12); ac = np.geomspace(1e11, 1e14, 12); imp = ac[::-1].copy(); so = np.linspace(0.0, 4e13, 12); params = (energy, weights, 0.05, ac, imp, so, 600.0, 1.5, 2.0, 8.0, 2.0)",
            "call": "compute_transport_moments(*params)",
            "gold_call": "_oracle_compute_transport_moments(*params)",
        },
        {
            "setup": "def bad_argument_sets():\n    good = (np.array([0.01, 0.05]), np.array([0.02, 0.02]), 0.03, np.array([1e13, 1e13]), np.zeros(2), np.zeros(2), 300.0, 0.405, 0.7, 4.0, 6.6)\n    bad = [list(good) for _ in range(15)]\n    bad[0][1] = np.array([0.02, 0.0])\n    bad[1][3] = np.array([np.nan, 1e13])\n    bad[2][3] = np.array([1e13, 1e13, 1e13])\n    bad[3][6] = -300.0\n    bad[4][3] = np.zeros(2)\n    bad[5][0] = np.array([0.0, 0.0])\n    bad[6][4] = np.array([-1.0e12, 0.0])\n    bad[7][8] = -0.1\n    bad[8][2] = np.nan\n    bad[9][0] = np.array([[0.01, 0.05]])\n    bad[10][0] = np.array([-0.01, 0.05])\n    bad[11][7] = 0.0\n    bad[12][9] = -4.0\n    bad[13][10] = 0.0\n    bad[14][0] = bad[14][1] = bad[14][3] = bad[14][4] = bad[14][5] = np.array([])\n    return bad\ndef rejected_public():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            compute_transport_moments(*args)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            _oracle_compute_transport_moments(*args)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
