"""
Compute screened surface-optical-phonon momentum relaxation.

For each mode use \(N_\nu=[\exp(\Omega_\nu/k_BT)-1]^{-1}\), \(k_\pm=k(E\pm\Omega_\nu)\), and \(q_\pm=(k^2+k_\pm^2-2kk_\pm\cos\phi)^{1/2}\). The angular kernel is



\[

\mathcal K(q,k')=\frac{\sinh^2(aq/2)[1-(k'/k)\cos\phi]}{q(4\pi^2q+a^2q^3)^2\varepsilon_{2D}(q)^2}.

\]



Use



\[

A_\nu=\frac{N_{int}32\pi^3e^2m^*}{\hbar^3a^2}\frac{\Omega_\nu}{2\epsilon_0}

\left[\frac{1}{2\kappa_\infty}-\frac{1}{\kappa_0+\kappa_\infty}\right]

\]



and sum the absorption and strictly allowed emission terms. The final-state density-of-states prefactor is the constant band-edge mass \(m^*\), with no \((1+2\alpha E_\pm)\) multiplier, and no final-state Pauli factor \(1-f(E_\pm)\) is included. Use \(\Theta(x)=1\) only for \(x>0\). Energies, angles, and mode energies must be nonempty one-dimensional arrays, or `ValueError` is raised. The summed rate is returned without clipping: a negative total at any energy raises `ValueError`, as does any momentum transfer with \(aq>700\).

Returns
-------
A NumPy vector containing the total surface-optical momentum-relaxation rate in s\(^{-1}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_surface_optical_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', mode_energy_mev: 'np.ndarray', absorption_q_m_inv: 'np.ndarray', emission_q_m_inv: 'np.ndarray', absorption_k_ratio: 'np.ndarray', emission_k_ratio: 'np.ndarray', absorption_dielectric: 'np.ndarray', emission_dielectric: 'np.ndarray', temperature_k: float, mass_ratio: float, thickness_angstrom: float, epsilon_static: float, epsilon_high_frequency: float, interface_count: int) -> 'np.ndarray':
    """Return the summed absorption and emission rates in s^-1.

    Use a constant band-edge final-state mass prefactor, without a Kane
    multiplier or a final-state Pauli occupation factor.

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Positive initial carrier energies in eV.
    scattering_angle_rad, angle_weights : numpy.ndarray
        Angular quadrature nodes and weights over zero to two pi.
    mode_energy_mev : numpy.ndarray
        Surface-optical mode energies in meV.
    absorption_q_m_inv, emission_q_m_inv : numpy.ndarray
        Momentum transfers with shape (mode, energy, angle).
    absorption_k_ratio, emission_k_ratio : numpy.ndarray
        Final-to-initial wave-vector ratios with shape (mode, energy).
    absorption_dielectric, emission_dielectric : numpy.ndarray
        Static carrier dielectric responses on the two inelastic grids.
    temperature_k, mass_ratio, thickness_angstrom : float
        Temperature, effective-mass ratio, and layer thickness.
    epsilon_static, epsilon_high_frequency : float
        Static and high-frequency permittivities of the dielectric.
    interface_count : int
        Number of identical dielectric interfaces contributing modes.

    Returns
    -------
    numpy.ndarray
        Total surface-optical momentum-relaxation rate in s^-1.

    Raises
    ------
    ValueError
        For empty, non-one-dimensional, or nonpositive energies; empty or
        non-one-dimensional angles, angles outside [0, 2*pi], or weights
        that are not positive and matching; empty, non-one-dimensional, or
        nonpositive mode energies; inconsistent grid shapes; nonfinite entries;
        nonpositive momentum transfers or dielectric responses; negative
        wave-vector ratios; nonpositive or nonfinite material inputs;
        epsilon_static below epsilon_high_frequency; an interface_count
        that is not a positive integer (booleans included); a momentum
        transfer with a * q above 700, where sinh^2(aq/2) overflows double
        precision; or supplied arrays for which the summed rate is negative
        at some energy (the rate is returned without clipping, so this can
        only come from angular factors 1 - (k'/k) cos(phi) that are not
        outweighed).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Compute screened surface-optical-phonon momentum relaxation."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31
_EPSILON_0 = 8.8541878128e-12

def _oracle_compute_surface_optical_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', mode_energy_mev: 'np.ndarray', absorption_q_m_inv: 'np.ndarray', emission_q_m_inv: 'np.ndarray', absorption_k_ratio: 'np.ndarray', emission_k_ratio: 'np.ndarray', absorption_dielectric: 'np.ndarray', emission_dielectric: 'np.ndarray', temperature_k: float, mass_ratio: float, thickness_angstrom: float, epsilon_static: float, epsilon_high_frequency: float, interface_count: int) -> 'np.ndarray':
    """Reference symmetric-interface surface-phonon rate."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _K_B = 1.380649e-23
    _M_E = 9.1093837015e-31
    _EPSILON_0 = 8.8541878128e-12
    energy = np.asarray(energy_ev, dtype=float)
    angle = np.asarray(scattering_angle_rad, dtype=float)
    weights = np.asarray(angle_weights, dtype=float)
    mode_mev = np.asarray(mode_energy_mev, dtype=float)
    q_abs = np.asarray(absorption_q_m_inv, dtype=float)
    q_em = np.asarray(emission_q_m_inv, dtype=float)
    ratio_abs = np.asarray(absorption_k_ratio, dtype=float)
    ratio_em = np.asarray(emission_k_ratio, dtype=float)
    dielectric_abs = np.asarray(absorption_dielectric, dtype=float)
    dielectric_em = np.asarray(emission_dielectric, dtype=float)
    if energy.ndim != 1 or energy.size == 0 or np.any(energy <= 0.0):
        raise ValueError('energy_ev must be a positive nonempty vector')
    if angle.ndim != 1 or weights.shape != angle.shape or angle.size == 0:
        raise ValueError('angle nodes and weights must be matching vectors')
    if not np.all(np.isfinite(angle)) or np.any(angle < 0.0) or np.any(angle > 2.0 * np.pi):
        raise ValueError('scattering angles must lie between zero and two pi')
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError('angle weights must be positive and finite')
    if mode_mev.ndim != 1 or mode_mev.size == 0 or np.any(mode_mev <= 0.0):
        raise ValueError('mode_energy_mev must be a positive nonempty vector')
    tensor_shape = (mode_mev.size, energy.size, angle.size)
    matrix_shape = (mode_mev.size, energy.size)
    if not (q_abs.shape == q_em.shape == dielectric_abs.shape == dielectric_em.shape == tensor_shape and ratio_abs.shape == ratio_em.shape == matrix_shape):
        raise ValueError('inelastic grids have inconsistent shapes')
    arrays = (energy, mode_mev, q_abs, q_em, ratio_abs, ratio_em, dielectric_abs, dielectric_em)
    if not all((np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError('surface-phonon inputs must be finite')
    if np.any(q_abs <= 0.0) or np.any(q_em <= 0.0):
        raise ValueError('momentum transfers must be positive')
    if np.any(ratio_abs < 0.0) or np.any(ratio_em < 0.0):
        raise ValueError('wave-vector ratios must be nonnegative')
    if np.any(dielectric_abs <= 0.0) or np.any(dielectric_em <= 0.0):
        raise ValueError('dielectric responses must be positive')
    values = (temperature_k, mass_ratio, thickness_angstrom, epsilon_static, epsilon_high_frequency)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite material inputs are required')
    if epsilon_static < epsilon_high_frequency:
        raise ValueError('static permittivity must not be below its high-frequency value')
    if isinstance(interface_count, (bool, np.bool_)) or not isinstance(interface_count, (int, np.integer)) or interface_count <= 0:
        raise ValueError('interface_count must be a positive integer')
    if max(np.max(q_abs), np.max(q_em)) * thickness_angstrom * 1e-10 > 700.0:
        raise ValueError('a times the momentum transfer must not exceed 700')
    mode_ev = mode_mev * 0.001
    mode_j = mode_ev * _E_CHARGE
    with np.errstate(over='ignore'):
        bose = 1.0 / np.expm1(mode_j / (_K_B * temperature_k))
    coupling_times_area = mode_j / (2.0 * _EPSILON_0) * (1.0 / (2.0 * epsilon_high_frequency) - 1.0 / (epsilon_static + epsilon_high_frequency))
    thickness_m = thickness_angstrom * 1e-10
    prefactor = interface_count * 32.0 * np.pi ** 3 * _E_CHARGE ** 2 * coupling_times_area * (mass_ratio * _M_E) / (_HBAR ** 3 * thickness_m ** 2)
    cosine = np.cos(angle)[None, None, :]

    def _kernel(momentum_transfer: np.ndarray) -> np.ndarray:
        aq = thickness_m * momentum_transfer
        return np.sinh(0.5 * aq) ** 2 / (momentum_transfer * (4.0 * np.pi ** 2 * momentum_transfer + thickness_m ** 2 * momentum_transfer ** 3) ** 2)
    absorption_integral = np.sum(weights[None, None, :] * _kernel(q_abs) * (1.0 - ratio_abs[:, :, None] * cosine) / dielectric_abs ** 2, axis=2)
    emission_integral = np.sum(weights[None, None, :] * _kernel(q_em) * (1.0 - ratio_em[:, :, None] * cosine) / dielectric_em ** 2, axis=2)
    emission_allowed = energy[None, :] > mode_ev[:, None]
    rate_by_mode = prefactor[:, None] * (bose[:, None] * absorption_integral + (bose + 1.0)[:, None] * emission_integral * emission_allowed)
    rate = np.sum(rate_by_mode, axis=0)
    if np.any(rate < 0.0):
        raise ValueError('the supplied grids give a negative total surface-optical rate')
    return rate

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "energy = np.array([0.03, 0.10]); angle = np.array([0.4, 2.2, 5.1]); weights = np.full(3, 2*np.pi/3); modes = np.array([20.0, 60.0]); shape = (2, 2, 3); qabs = np.full(shape, 8e8); qem = np.full(shape, 6e8); rabs = np.full((2, 2), 1.2); rem = np.full((2, 2), 0.7); epsa = np.full(shape, 2.0); epse = np.full(shape, 1.8); params = (energy, angle, weights, modes, qabs, qem, rabs, rem, epsa, epse, 300.0, 0.405, 6.6, 23.0, 5.03, 2)",
            "call": "compute_surface_optical_rates(*params)",
            "gold_call": "_oracle_compute_surface_optical_rates(*params)",
        },
        {
            "setup": "energy = np.array([1e-6]); angle = np.array([np.pi]); weights = np.array([2*np.pi]); modes = np.array([200.0]); shape = (1, 1, 1); qabs = np.full(shape, 1e9); qem = np.full(shape, 1e8); ratio = np.ones((1, 1)); dielectric = np.ones(shape); params = (energy, angle, weights, modes, qabs, qem, ratio, ratio, dielectric, dielectric, 80.0, 0.405, 2.0, 4.0, 2.0, 1)",
            "call": "compute_surface_optical_rates(*params)",
            "gold_call": "_oracle_compute_surface_optical_rates(*params)",
        },
        {
            "setup": "energy = np.array([0.01, 0.04]); angle = np.array([0.2, 3.0]); weights = np.array([np.pi, np.pi]); modes = np.array([10.0]); shape = (1, 2, 2); q = np.full(shape, 5e8); ratio = np.full((1, 2), 0.8); dielectric = np.full(shape, 3.0); params = (energy, angle, weights, modes, q, q, ratio, ratio, dielectric, dielectric, 300.0, 1.0, 6.6, 6.0, 4.0, 2)",
            "call": "compute_surface_optical_rates(*params)",
            "gold_call": "_oracle_compute_surface_optical_rates(*params)",
        },
        {
            "setup": "def bad_argument_sets():\n    good = (np.array([0.03, 0.10]), np.array([1.0, 4.0]), np.array([3.0, 3.283185307179586]), np.array([20.0]), np.full((1, 2, 2), 8.0e8), np.full((1, 2, 2), 6.0e8), np.full((1, 2), 1.2), np.full((1, 2), 0.7), np.full((1, 2, 2), 2.0), np.full((1, 2, 2), 1.8), 300.0, 0.405, 6.6, 23.0, 5.03, 2)\n    bad = [list(good) for _ in range(33)]\n    bad[0][0] = np.array([0.0, 0.10])\n    bad[1][3] = np.array([-20.0])\n    bad[2][4] = np.zeros((1, 2, 2))\n    bad[3][6] = np.full((1, 2), -1.2)\n    bad[4][13] = 4.0\n    bad[5][15] = True\n    bad[6][15] = 0\n    bad[7][1] = np.array([0.1, 6.2])\n    bad[7][6] = np.full((1, 2), 3.0)\n    bad[8][1] = np.array([1.0, 7.0])\n    bad[9][2] = np.array([3.0, 3.0, 0.283185307179586])\n    bad[10][4] = np.full((1, 2, 3), 8.0e8)\n    bad[11][4] = np.full((1, 2, 2), np.nan)\n    bad[12][10] = np.nan\n    bad[13][15] = 2.5\n    bad[14][4] = np.full((1, 2, 2), 2.0e12)\n    bad[15][0] = np.array([])\n    bad[15][4] = np.full((1, 0, 2), 8.0e8)\n    bad[15][5] = np.full((1, 0, 2), 6.0e8)\n    bad[15][6] = np.full((1, 0), 1.2)\n    bad[15][7] = np.full((1, 0), 0.7)\n    bad[15][8] = np.full((1, 0, 2), 2.0)\n    bad[15][9] = np.full((1, 0, 2), 1.8)\n    bad[16][2] = np.array([3.0, -3.283185307179586])\n    bad[17][3] = np.array([])\n    bad[17][4] = np.full((0, 2, 2), 8.0e8)\n    bad[17][5] = np.full((0, 2, 2), 6.0e8)\n    bad[17][6] = np.full((0, 2), 1.2)\n    bad[17][7] = np.full((0, 2), 0.7)\n    bad[17][8] = np.full((0, 2, 2), 2.0)\n    bad[17][9] = np.full((0, 2, 2), 1.8)\n    bad[18][12] = 0.0\n    bad[19][1] = np.array([])\n    bad[19][2] = np.array([])\n    bad[19][4] = np.full((1, 2, 0), 8.0e8)\n    bad[19][5] = np.full((1, 2, 0), 6.0e8)\n    bad[19][8] = np.full((1, 2, 0), 2.0)\n    bad[19][9] = np.full((1, 2, 0), 1.8)\n    bad[20][0] = np.array([[0.03, 0.10]])\n    bad[21][1] = np.array([[1.0, 4.0]])\n    bad[21][2] = np.array([[3.0, 3.283185307179586]])\n    bad[22][3] = np.array([[20.0]])\n    bad[23][8] = np.zeros((1, 2, 2))\n    bad[24][10] = 0.0\n    bad[25][11] = -0.405\n    bad[26][13] = 0.0\n    bad[27][14] = 0.0\n    bad[28][11] = np.nan\n    bad[29][12] = np.inf\n    bad[30][13] = np.nan\n    bad[31][14] = np.inf\n    bad[32][5] = np.full((1, 2, 2), 2.0e12)\n    return bad\ndef rejected_public():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            compute_surface_optical_rates(*args)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for args in bad_argument_sets():\n        try:\n            _oracle_compute_surface_optical_rates(*args)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
