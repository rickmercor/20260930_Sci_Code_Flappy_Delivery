"""
Evaluate the thermally broadened two-dimensional polarizability.

Thermally average the zero-temperature response over the supplied energy range, with Gauss-Legendre quadrature of the supplied order:

$$
\Pi(q,T)=\int_0^{E_{max}}\frac{\Pi_0(q;E)}{4k_BT\cosh^2[(\mu-E)/(2k_BT)]}\,dE,
$$

$$
\Pi_0(q;E)=\frac{g m^*}{2\pi\hbar^2}
\begin{cases}
1,&q\le 2k(E),\\
1-\sqrt{1-[2k(E)/q]^2},&q>2k(E).
  \end{cases}
$$

The prefactor is the constant band-edge value $g m^*/(2\pi\hbar^2)$, with no nonparabolic $(1+2\alpha E)$ multiplier. The supplied momentum coordinates are $q/k_F$.

Returns
-------
A NumPy array with the input shape containing \(\Pi(q,T)\) in J\(^{-1}\) m\(^{-2}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_polarizability(q_over_kf: 'np.ndarray', k_fermi_m_inv: float, temperature_k: float, chemical_potential_ev: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Return finite-temperature static polarizability.

    The zero-temperature response uses the constant band-edge prefactor
    g*m/(2*pi*hbar^2), without a Kane (1 + 2*alpha*E) multiplier.

    Parameters
    ----------
    q_over_kf : numpy.ndarray
        Positive scattering wave vectors divided by k_F.
    k_fermi_m_inv : float
        Fermi wave vector in m^-1 from the band-occupation stage.
    temperature_k : float
        Temperature in kelvin.
    chemical_potential_ev : float
        Chemical potential relative to the band edge in eV.
    mass_ratio : float
        Effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.
    degeneracy : float
        Combined spin and valley degeneracy.
    energy_max_ev : float
        Upper thermal-average energy in eV.
    quadrature_order : int
        Gauss-Legendre order for the thermal average.

    Returns
    -------
    numpy.ndarray
        Polarizability in J^-1 m^-2 with the input shape.

    Raises
    ------
    ValueError
        For empty, nonfinite, or nonpositive q_over_kf; nonfinite or
        nonpositive k_fermi_m_inv, temperature, mass ratio, degeneracy, or
        energy maximum; a nonfinite chemical potential; a negative or
        nonfinite alpha_ev_inv; or a quadrature order that is not a
        positive integer (booleans included).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Evaluate the thermally broadened two-dimensional polarizability."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def _oracle_evaluate_polarizability(q_over_kf: 'np.ndarray', k_fermi_m_inv: float, temperature_k: float, chemical_potential_ev: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Reference thermal average of the zero-temperature 2D response."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _K_B = 1.380649e-23
    _M_E = 9.1093837015e-31
    q_ratio = np.asarray(q_over_kf, dtype=float)
    if q_ratio.size == 0 or not np.all(np.isfinite(q_ratio)) or np.any(q_ratio <= 0.0):
        raise ValueError('q_over_kf must contain positive finite values')
    values = (k_fermi_m_inv, temperature_k, mass_ratio, degeneracy, energy_max_ev)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite band and integration inputs are required')
    if not np.isfinite(chemical_potential_ev):
        raise ValueError('chemical_potential_ev must be finite')
    if not np.isfinite(alpha_ev_inv) or alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be finite and nonnegative')
    if isinstance(quadrature_order, (bool, np.bool_)) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order <= 0:
        raise ValueError('quadrature_order must be a positive integer')
    shape = q_ratio.shape
    q_flat = q_ratio.reshape(-1) * k_fermi_m_inv
    nodes, weights = leggauss(quadrature_order)
    energy_ev = 0.5 * energy_max_ev * (nodes + 1.0)
    weights_j = 0.5 * energy_max_ev * weights * _E_CHARGE
    energy_j = energy_ev * _E_CHARGE
    mass = mass_ratio * _M_E
    k_energy = np.sqrt(2.0 * mass * energy_j * (1.0 + alpha_ev_inv * energy_ev)) / _HBAR
    occupation = expit((chemical_potential_ev - energy_ev) * _E_CHARGE / (_K_B * temperature_k))
    thermal_weight = occupation * (1.0 - occupation) / (_K_B * temperature_k)
    density_scale = degeneracy * mass / (2.0 * np.pi * _HBAR ** 2)
    result = np.empty_like(q_flat)
    weighted_thermal = thermal_weight * weights_j
    for start in range(0, q_flat.size, quadrature_order):
        q_block = q_flat[start:start + quadrature_order]
        ratio = 2.0 * k_energy[None, :] / q_block[:, None]
        zero_temperature_shape = np.ones_like(ratio)
        outside_disk = ratio < 1.0
        zero_temperature_shape[outside_disk] -= np.sqrt(1.0 - ratio[outside_disk] ** 2)
        result[start:start + q_block.size] = zero_temperature_shape @ weighted_thermal
    return (density_scale * result).reshape(shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection differential tests."""
    return [{'setup': 'ratio = np.array([0.02, 0.5, 2.0, 6.0]); params = (ratio, 7.3e8, 300.0, 0.045, 0.405, '
               '0.7, 4.0, 1.0, 180)',
      'call': 'evaluate_polarizability(*params) / 1e37',
      'gold_call': '_oracle_evaluate_polarizability(*params) / 1e37',
      'tol': 1e-09},
     {'setup': 'ratio = np.array([1e-8]); params = (ratio, 1.0e8, 300.0, -0.05, 0.405, 0.0, 2.0, 0.8, '
               '120)',
      'call': 'evaluate_polarizability(*params) / 1e37',
      'gold_call': '_oracle_evaluate_polarizability(*params) / 1e37',
      'tol': 1e-09},
     {'setup': 'ratio = np.array([[0.1, 1.0], [4.0, 12.0]]); params = (ratio, 8.0e8, 40.0, 0.08, 1.2, '
               '1.5, 6.0, 1.2, 160)',
      'call': 'evaluate_polarizability(*params) / 1e37',
      'gold_call': '_oracle_evaluate_polarizability(*params) / 1e37',
      'tol': 1e-09},
     {'setup': 'def bad_argument_sets():\n'
               '    good = (np.array([0.5, 2.0]), 8.0e8, 300.0, 0.03, 0.405, 0.7, 4.0, 1.0, 20)\n'
               '    bad = [list(good) for _ in range(19)]\n'
               '    bad[0][0] = np.array([0.0, 2.0])\n'
               '    bad[1][1] = 0.0\n'
               '    bad[2][2] = np.nan\n'
               '    bad[3][5] = -1.0\n'
               '    bad[4][8] = 1.5\n'
               '    bad[5][8] = False\n'
               '    bad[6][0] = np.array([np.nan, 2.0])\n'
               '    bad[7][3] = np.nan\n'
               '    bad[8][8] = 0\n'
               '    bad[9][0] = np.array([])\n'
               '    bad[10][5] = np.nan\n'
               '    bad[11][1] = np.inf\n'
               '    bad[12][4] = np.nan\n'
               '    bad[13][6] = np.nan\n'
               '    bad[14][7] = np.inf\n'
               '    bad[15][2] = -1.0\n'
               '    bad[16][4] = 0.0\n'
               '    bad[17][6] = -4.0\n'
               '    bad[18][7] = 0.0\n'
               '    return bad\n'
               'def rejected_public():\n'
               '    count = 0.0\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            evaluate_polarizability(*args)\n'
               '        except ValueError:\n'
               '            count += 1.0\n'
               '    return count\n'
               'def rejected_oracle():\n'
               '    count = 0.0\n'
               '    for args in bad_argument_sets():\n'
               '        try:\n'
               '            _oracle_evaluate_polarizability(*args)\n'
               '        except ValueError:\n'
               '            count += 1.0\n'
               '    return count',
      'call': 'rejected_public()',
      'gold_call': 'rejected_oracle()',
      'tol': 1e-09}]
