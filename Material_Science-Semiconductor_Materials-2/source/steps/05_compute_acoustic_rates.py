"""
Compute incidence-averaged, partially screened acoustic momentum relaxation.

For elastic momentum transfer \(q=2k(E)\sin(\phi/2)\) at scattering-angle node \(\phi_i\) with weight \(w_i\), the supplied array holds four rows of band-edge shifts in eV at every scattering node \(i\) and incidence node \(j\) (weight \(u_j\)): an LA shift \(S^{LA}_{ij}\) that is divided by the dielectric response, an LA shift \(U^{LA}_{ij}\) that is not, and the TA pair \(S^{TA}_{ij}\) and \(U^{TA}_{ij}\) in the same roles. With \(\varepsilon_{ni}\) the response at energy node \(n\),



\[

D^{LA}_{nij}=\frac{S^{LA}_{ij}}{\varepsilon_{ni}}+U^{LA}_{ij},\qquad

D^{TA}_{nij}=\frac{S^{TA}_{ij}}{\varepsilon_{ni}}+U^{TA}_{ij},

\]



\[

\Gamma_{ac}(E_n)=\frac{m^*k_BT(1+2\alpha E_n)}{2\pi\hbar^3}

\sum_i w_i(1-\cos\phi_i)\sum_j\frac{u_j}{2\pi}\left[\frac{(D^{LA}_{nij})^2}{c_{11}}+\frac{(D^{TA}_{nij})^2}{c_{66}}\right],\qquad c_{66}=\frac{c_{11}-c_{12}}{2},

\]



with the shifts converted from eV to J and the incidence weights used as given. The prefactor uses the initial-state Kane density-of-states factor \(m^*(1+2\alpha E)\).

Returns
-------
A NumPy vector of acoustic momentum-relaxation rates in s\(^{-1}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_acoustic_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', incidence_weights: 'np.ndarray', band_shift_ev: 'np.ndarray', dielectric_response: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, c11_n_m: float, c12_n_m: float) -> 'np.ndarray':
    """Return incidence-averaged acoustic momentum-relaxation rates.

    The four rows of band_shift_ev hold, for every scattering-angle node i
    and incidence node j, the LA shift that is divided by the dielectric
    response (row 0), the LA shift that is not (row 1), the TA shift that is
    divided by the dielectric response (row 2), and the TA shift that is not
    (row 3), all in eV. With eps = dielectric_response[n, i] at energy node
    n, the event couplings are

        D_LA = row0 / eps + row1,    D_TA = row2 / eps + row3,

    and the rate is

        Gamma(E_n) = m* k_B T (1 + 2 alpha E_n) / (2 pi hbar^3)
                     * sum_i w_i (1 - cos phi_i)
                       * sum_j u_j / (2 pi) * [D_LA^2 / c11 + D_TA^2 / c66],

    with c66 = (c11 - c12) / 2, w_i = angle_weights, u_j =
    incidence_weights, and the shifts converted from eV to J.

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Nonempty one-dimensional array of nonnegative carrier energies in eV.
    scattering_angle_rad, angle_weights : numpy.ndarray
        One-dimensional scattering-angle nodes phi_i and positive weights of
        the same length.
    incidence_weights : numpy.ndarray
        Nonempty one-dimensional positive weights u_j of the average over the
        incident direction; their sum is used as given.
    band_shift_ev : numpy.ndarray
        Shape (4, n_angle, n_incidence), the four shift rows defined above.
    dielectric_response : numpy.ndarray
        Positive dimensionless response with shape (n_energy, n_angle).
    temperature_k, mass_ratio : float
        Positive temperature in kelvin and effective mass divided by the
        electron mass.
    alpha_ev_inv : float
        Nonnegative nonparabolicity in eV^-1.
    c11_n_m, c12_n_m : float
        Two-dimensional elastic constants in N m^-1 with c11 > c12 >= 0.

    Returns
    -------
    numpy.ndarray
        Acoustic momentum-relaxation rates in s^-1, one per energy node.

    Raises
    ------
    ValueError
        For an empty or non-one-dimensional energy or incidence-weight
        array, a non-one-dimensional angle array, angle weights not matching
        the angles, a shift array whose shape is not
        (4, n_angle, n_incidence), a dielectric shape other than
        (n_energy, n_angle), nonfinite entries, negative energies,
        nonpositive angle or incidence weights or dielectric response,
        nonpositive or nonfinite temperature or mass, a negative or
        nonfinite alpha_ev_inv, or nonfinite elastic constants or elastic
        constants violating c11 > c12 >= 0.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Compute incidence-averaged, partially screened acoustic momentum relaxation."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def _oracle_compute_acoustic_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', incidence_weights: 'np.ndarray', band_shift_ev: 'np.ndarray', dielectric_response: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, c11_n_m: float, c12_n_m: float) -> 'np.ndarray':
    """Reference incidence-averaged acoustic angular integral."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _K_B = 1.380649e-23
    _M_E = 9.1093837015e-31
    energy = np.asarray(energy_ev, dtype=float)
    angle = np.asarray(scattering_angle_rad, dtype=float)
    weights = np.asarray(angle_weights, dtype=float)
    incidence = np.asarray(incidence_weights, dtype=float)
    shifts = np.asarray(band_shift_ev, dtype=float)
    dielectric = np.asarray(dielectric_response, dtype=float)
    if energy.ndim != 1 or energy.size == 0 or angle.ndim != 1:
        raise ValueError('energy and scattering angle must be vectors and energy nonempty')
    if incidence.ndim != 1 or incidence.size == 0:
        raise ValueError('incidence_weights must be a nonempty vector')
    if weights.shape != angle.shape or shifts.shape != (4, angle.size, incidence.size):
        raise ValueError('angle weights or band shifts have incompatible shape')
    if dielectric.shape != (energy.size, angle.size):
        raise ValueError('dielectric_response has an incompatible shape')
    arrays = (energy, angle, weights, incidence, shifts, dielectric)
    if not all((np.all(np.isfinite(array)) for array in arrays)):
        raise ValueError('all acoustic arrays must contain finite values')
    if np.any(energy < 0.0) or np.any(weights <= 0.0) or np.any(incidence <= 0.0):
        raise ValueError('energy must be nonnegative and weights positive')
    if np.any(dielectric <= 0.0):
        raise ValueError('dielectric response must be positive')
    values = (temperature_k, mass_ratio)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite temperature and mass are required')
    if not np.isfinite(alpha_ev_inv) or alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be finite and nonnegative')
    if not all(np.isfinite((c11_n_m, c12_n_m))) or c11_n_m <= c12_n_m or c12_n_m < 0.0:
        raise ValueError('elastic constants must be finite with c11 > c12 >= 0')
    c66_n_m = 0.5 * (c11_n_m - c12_n_m)
    inverse_eps = 1.0 / dielectric
    longitudinal = (shifts[0][None, :, :] * inverse_eps[:, :, None] + shifts[1][None, :, :]) * _E_CHARGE
    transverse = (shifts[2][None, :, :] * inverse_eps[:, :, None] + shifts[3][None, :, :]) * _E_CHARGE
    coupling = longitudinal ** 2 / c11_n_m + transverse ** 2 / c66_n_m
    incidence_average = coupling @ incidence / (2.0 * np.pi)
    momentum_weight = 1.0 - np.cos(angle)
    contracted = incidence_average * momentum_weight[None, :] @ weights / (2.0 * np.pi)
    prefactor = mass_ratio * _M_E * _K_B * temperature_k / _HBAR ** 3
    return prefactor * (1.0 + 2.0 * alpha_ev_inv * energy) * contracted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection differential tests."""
    return [{'setup': 'energy = np.array([0.01, 0.05, 0.18])\n'
               'angle = np.array([0.4, 1.5, 3.4, 5.8])\n'
               'weights = np.array([1.0, 1.7, 2.0, 2.0*np.pi - 4.7])\n'
               'incidence = np.array([1.1, 2.0, 2.0*np.pi - 3.1])\n'
               'idx = np.arange(48).reshape(4, 4, 3)\n'
               'shifts = 2.0*np.sin(0.6*idx) + 0.7*np.cos(0.3*idx)\n'
               'dielectric = np.array([[2.0, 1.5, 3.0, 1.2], [1.8, 2.2, 1.1, 3.5], [1.2, 1.4, 2.2, '
               '1.7]])\n'
               'params = (energy, angle, weights, incidence, shifts, dielectric, 300.0, 0.405, 0.7, '
               '132.7, 33.0)',
      'call': 'compute_acoustic_rates(*params) / 1e12',
      'gold_call': '_oracle_compute_acoustic_rates(*params) / 1e12',
      'tol': 1e-09},
     {'setup': 'energy = np.array([0.0, 0.02])\n'
               'angle = np.array([0.7, 2.1, 4.5])\n'
               'weights = np.array([1.0, 2.0, 3.0])\n'
               'incidence = np.array([0.5, 1.5])\n'
               'shifts = np.empty((4, 3, 2))\n'
               'shifts[0] = 3.0\n'
               'shifts[1] = -3.0\n'
               'shifts[2] = 0.5\n'
               'shifts[3] = -0.5\n'
               'params = (energy, angle, weights, incidence, shifts, np.ones((2, 3)), 80.0, 0.8, 0.0, '
               '120.0, 0.0)',
      'call': 'compute_acoustic_rates(*params) / 1e12',
      'gold_call': '_oracle_compute_acoustic_rates(*params) / 1e12',
      'tol': 1e-09},
     {'setup': 'energy = np.array([0.0, 0.12])\n'
               'angle = np.array([2.0*np.pi/3.0])\n'
               'weights = np.array([1.7])\n'
               'incidence = np.array([0.7, 1.2])\n'
               'shifts = np.array([[[0.0, 0.0]], [[2.0, -0.5]], [[0.0, 0.0]], [[-1.0, 0.8]]])\n'
               'params = (energy, angle, weights, incidence, shifts, np.array([[1.0], [7.0]]), 40.0, '
               '1.2, 1.5, 140.0, 60.0)',
      'call': 'compute_acoustic_rates(*params) / 1e12',
      'gold_call': '_oracle_compute_acoustic_rates(*params) / 1e12',
      'tol': 1e-09},
     {'setup': 'good = (np.array([0.01, 0.05]), np.array([1.0, 4.0]), np.array([3.0, 3.0]), '
               'np.array([2.0, 4.0]), np.ones((4, 2, 2)), np.full((2, 2), 2.0), 300.0, 0.405, 0.7, '
               '132.7, 33.0)\n'
               'def bad_argument_sets():\n'
               '    bad = [list(good) for _ in range(18)]\n'
               '    bad[0][0] = np.array([])\n'
               '    bad[1][0] = np.array([[0.01, 0.05]])\n'
               '    bad[2][1] = np.array([[1.0, 4.0]])\n'
               '    bad[3][2] = np.array([3.0])\n'
               '    bad[4][3] = np.array([])\n'
               '    bad[5][4] = np.ones((4, 2, 3))\n'
               '    bad[6][5] = np.ones((2, 3))\n'
               '    bad[7][0] = np.array([-0.01, 0.05])\n'
               '    bad[8][2] = np.array([3.0, 0.0])\n'
               '    bad[9][3] = np.array([2.0, -4.0])\n'
               '    bad[10][4] = np.full((4, 2, 2), np.nan)\n'
               '    bad[11][5] = np.zeros((2, 2))\n'
               '    bad[12][6] = 0.0\n'
               '    bad[13][7] = np.inf\n'
               '    bad[14][8] = -0.1\n'
               '    bad[15][8] = np.nan\n'
               '    bad[16][9] = 33.0\n'
               '    bad[17][10] = -1.0\n'
               '    return bad\n'
               'def rejected_public():\n'
               '    result = []\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            compute_acoustic_rates(*args)\n'
               '        except ValueError:\n'
               '            result.append(1.0)\n'
               '        else:\n'
               '            result.append(0.0)\n'
               '    return np.asarray(result)\n'
               'def rejected_oracle():\n'
               '    result = []\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            _oracle_compute_acoustic_rates(*args)\n'
               '        except ValueError:\n'
               '            result.append(1.0)\n'
               '        else:\n'
               '            result.append(0.0)\n'
               '    return np.asarray(result)\n',
      'call': 'rejected_public()',
      'gold_call': 'rejected_oracle()',
      'tol': 1e-09}]
