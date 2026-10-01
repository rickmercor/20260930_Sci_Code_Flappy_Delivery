#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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

def solve_kane_state(temperature_k: float, density_cm2: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
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

"""Contract an in-plane deformation-potential tensor into LA and TA band-edge shifts."""
import numpy as np

def resolve_acoustic_tensor(phonon_angle_rad: 'np.ndarray', dp_tensor_ev: 'Sequence[float]') -> 'np.ndarray':
    """Reference longitudinal and transverse contraction of an in-plane tensor."""
    try:
        angle = np.asarray(phonon_angle_rad, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError('phonon_angle_rad must hold finite numbers') from error
    if angle.size == 0 or not np.all(np.isfinite(angle)):
        raise ValueError('phonon_angle_rad must be a nonempty array of finite values')
    try:
        tensor = np.asarray(dp_tensor_ev, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('dp_tensor_ev must hold three finite numbers') from error
    if tensor.shape != (3,) or not np.all(np.isfinite(tensor)):
        raise ValueError('dp_tensor_ev must hold three finite numbers')
    xi_xx, xi_yy, xi_xy = tensor
    sine = np.sin(angle)
    cosine = np.cos(angle)
    longitudinal_ev = xi_xx * cosine ** 2 + xi_yy * sine ** 2 + 2.0 * xi_xy * sine * cosine
    transverse_ev = (xi_yy - xi_xx) * sine * cosine + xi_xy * (cosine ** 2 - sine ** 2)
    return np.vstack((longitudinal_ev, transverse_ev))

"""Evaluate the thermally broadened two-dimensional polarizability."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def evaluate_polarizability(q_over_kf: 'np.ndarray', k_fermi_m_inv: float, temperature_k: float, chemical_potential_ev: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
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

"""Apply closed-form image-charge form factors and free-carrier screening."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_EPSILON_0 = 8.8541878128e-12

def screen_dielectric_kernel(q_m_inv: 'np.ndarray', polarizability_j_inv_m2: 'np.ndarray', thickness_angstrom: float, epsilon_layer: float, epsilon_environment: float) -> 'np.ndarray':
    """Reference infinite-image dielectric kernel with closed-form profile integrals."""
    _E_CHARGE = 1.602176634e-19
    _EPSILON_0 = 8.8541878128e-12
    q_input = np.asarray(q_m_inv, dtype=float)
    polarizability = np.asarray(polarizability_j_inv_m2, dtype=float)
    if q_input.shape != polarizability.shape or q_input.size == 0:
        raise ValueError('q and polarizability must be nonempty arrays with matching shapes')
    if not np.all(np.isfinite(q_input)) or np.any(q_input <= 0.0):
        raise ValueError('q_m_inv must contain positive finite values')
    if not np.all(np.isfinite(polarizability)) or np.any(polarizability < 0.0):
        raise ValueError('polarizability must contain finite nonnegative values')
    values = (thickness_angstrom, epsilon_layer, epsilon_environment)
    if not all(np.isfinite(values)) or min(values) <= 0.0:
        raise ValueError('positive finite dielectric inputs are required')
    thickness_m = thickness_angstrom * 1e-10
    x = q_input * thickness_m
    if np.any(x > 700.0):
        raise ValueError('q times the layer thickness must not exceed 700')
    four_pi_sq = 4.0 * np.pi ** 2
    denominator = x ** 2 + four_pi_sq
    direct_impurity = -2.0 * np.expm1(-0.5 * x) / x + 2.0 * x * (1.0 + np.exp(-0.5 * x)) / denominator
    projected_cosh = 2.0 * np.pi ** 2 * (2.0 * np.sinh(0.5 * x) / x) * 2.0 / denominator
    small = x < 0.01
    g = np.empty_like(x)
    xs = x[small]
    g[small] = 0.5 - xs / 6.0 + xs ** 2 / 24.0 - xs ** 3 / 120.0 + xs ** 4 / 720.0
    xl = x[~small]
    g[~small] = (xl + np.expm1(-xl)) / xl ** 2
    direct_carrier = (3.0 * x + (8.0 * np.pi ** 2 * x + 32.0 * np.pi ** 4 * g) / denominator) / denominator
    contrast = (epsilon_layer - epsilon_environment) / (epsilon_layer + epsilon_environment)
    image_ratio = contrast * np.exp(-x)
    image_sum = 2.0 * image_ratio / (1.0 - image_ratio)
    impurity_form = direct_impurity + image_sum * projected_cosh
    carrier_form = direct_carrier + image_sum * projected_cosh ** 2
    coulomb_prefactor = _E_CHARGE ** 2 / (2.0 * _EPSILON_0 * epsilon_layer * q_input)
    dielectric = 1.0 + coulomb_prefactor * polarizability * carrier_form
    screened_potential = coulomb_prefactor * impurity_form / dielectric
    return np.stack((screened_potential, dielectric), axis=0)

"""Compute incidence-averaged, partially screened acoustic momentum relaxation."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def compute_acoustic_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', incidence_weights: 'np.ndarray', band_shift_ev: 'np.ndarray', dielectric_response: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, c11_n_m: float, c12_n_m: float) -> 'np.ndarray':
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

"""Integrate the screened Coulomb rate coefficient over scattering angle."""
import numpy as np
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def integrate_impurity_rate_coefficient(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', screened_potential_j_m2: 'np.ndarray', mass_ratio: float, alpha_ev_inv: float) -> 'np.ndarray':
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

"""Compute screened surface-optical-phonon momentum relaxation."""
import numpy as np
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31
_EPSILON_0 = 8.8541878128e-12

def compute_surface_optical_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', mode_energy_mev: 'np.ndarray', absorption_q_m_inv: 'np.ndarray', emission_q_m_inv: 'np.ndarray', absorption_k_ratio: 'np.ndarray', emission_k_ratio: 'np.ndarray', absorption_dielectric: 'np.ndarray', emission_dielectric: 'np.ndarray', temperature_k: float, mass_ratio: float, thickness_angstrom: float, epsilon_static: float, epsilon_high_frequency: float, interface_count: int) -> 'np.ndarray':
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

"""Convert energy-resolved relaxation rates into thermoelectric and Hall responses."""
import numpy as np
from scipy.special import expit
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_K_B = 1.380649e-23
_M_E = 9.1093837015e-31

def compute_transport_moments(energy_ev: 'np.ndarray', energy_weights_ev: 'np.ndarray', chemical_potential_ev: float, acoustic_rate_s_inv: 'np.ndarray', impurity_rate_s_inv: 'np.ndarray', surface_optical_rate_s_inv: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, thickness_angstrom: float) -> 'np.ndarray':
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

"""Infer the gate-held sheet density and midplane defect density from Hall-bar data."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def infer_sheet_state(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), epsilon_static: 'float'=3.9, epsilon_high_frequency: 'float'=2.5, mode_energy_mev: 'Sequence[float]'=(55.6, 138.1), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> 'np.ndarray':
    """Reference least-squares inference through the full transport chain."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _M_E = 9.1093837015e-31
    solve_kane_state_stage = solve_kane_state
    resolve_acoustic_tensor_stage = resolve_acoustic_tensor
    evaluate_polarizability_stage = evaluate_polarizability
    screen_dielectric_kernel_stage = screen_dielectric_kernel
    compute_acoustic_rates_stage = compute_acoustic_rates
    integrate_impurity_rate_coefficient_stage = integrate_impurity_rate_coefficient
    compute_surface_optical_rates_stage = compute_surface_optical_rates
    compute_transport_moments_stage = compute_transport_moments

    def _positive_integer(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(name + ' must be a positive integer')
        return int(value)
    temperatures = np.asarray(temperatures_k, dtype=float)
    resistances = np.asarray(sheet_resistance_ohm, dtype=float)
    slopes = np.asarray(hall_slope_ohm_per_tesla, dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 2:
        raise ValueError('at least two temperatures are required')
    if resistances.shape != temperatures.shape or slopes.shape != temperatures.shape:
        raise ValueError('each temperature needs one sheet resistance and one Hall slope')
    for array in (temperatures, resistances, slopes):
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError('temperatures and Hall-bar data must be positive and finite')
    if np.unique(temperatures).size != temperatures.size:
        raise ValueError('temperatures must be distinct')
    modes_mev = np.asarray(mode_energy_mev, dtype=float)
    if modes_mev.ndim != 1 or modes_mev.size == 0 or (not np.all(np.isfinite(modes_mev))) or np.any(modes_mev <= 0.0):
        raise ValueError('mode_energy_mev must be a nonempty vector of positive energies')
    bounds = np.asarray((density_bounds_cm2, impurity_bounds_cm2), dtype=float)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError('both search intervals must be finite pairs')
    if np.any(bounds[:, 0] <= 0.0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('search intervals must satisfy 0 < lower < upper')
    tensor_pairs = []
    for value, name in ((screened_dp_tensors_ev, 'screened_dp_tensors_ev'), (unscreened_dp_tensors_ev, 'unscreened_dp_tensors_ev')):
        try:
            pair = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(name + ' must be a (2, 3) array of finite numbers') from error
        if pair.shape != (2, 3) or not np.all(np.isfinite(pair)):
            raise ValueError(name + ' must be a (2, 3) array of finite numbers')
        tensor_pairs.append(pair)
    screened_pair, unscreened_pair = tensor_pairs
    scalars = np.asarray((epsilon_static, epsilon_high_frequency, mass_ratio, alpha_ev_inv, degeneracy, c11_n_m, c12_n_m, thickness_angstrom, epsilon_layer, transport_energy_max_ev, polarizability_energy_max_ev), dtype=float)
    if not np.all(np.isfinite(scalars)):
        raise ValueError('material parameters must be finite')
    if epsilon_high_frequency <= 0.0 or epsilon_static < epsilon_high_frequency:
        raise ValueError('permittivities must satisfy epsilon_static >= epsilon_high_frequency > 0')
    if min(mass_ratio, degeneracy, thickness_angstrom, epsilon_layer) <= 0.0:
        raise ValueError('mass, degeneracy, thickness, and layer permittivity must be positive')
    if alpha_ev_inv < 0.0:
        raise ValueError('alpha_ev_inv must be nonnegative')
    if c11_n_m <= c12_n_m or c12_n_m < 0.0:
        raise ValueError('elastic constants must satisfy c11 > c12 >= 0')
    if transport_energy_max_ev <= 1e-06 or polarizability_energy_max_ev <= 0.0:
        raise ValueError('energy maxima must be positive and the transport maximum above 1e-6 eV')
    if np.any(modes_mev * 0.001 >= transport_energy_max_ev):
        raise ValueError('mode energies must lie below the transport maximum')
    orders = [_positive_integer(value, name) for value, name in ((density_order, 'density_order'), (transport_order_per_segment, 'transport_order_per_segment'), (angle_order, 'angle_order'), (polarizability_order, 'polarizability_order'))]
    density_order, transport_order_per_segment, angle_order, polarizability_order = orders
    interfaces = _positive_integer(interface_count, 'interface_count')
    angle_nodes, angle_weights = leggauss(angle_order)
    scattering_angle = np.pi * (angle_nodes + 1.0)
    angle_weights = np.pi * angle_weights
    phonon_angle = scattering_angle[None, :] + 0.5 * scattering_angle[:, None] + 0.5 * np.pi
    band_shift = np.stack((resolve_acoustic_tensor_stage(phonon_angle, screened_pair[0])[0], resolve_acoustic_tensor_stage(phonon_angle, unscreened_pair[0])[0], resolve_acoustic_tensor_stage(phonon_angle, screened_pair[1])[1], resolve_acoustic_tensor_stage(phonon_angle, unscreened_pair[1])[1])).reshape(4, angle_order, angle_order)
    modes_ev = modes_mev * 0.001
    boundaries = np.concatenate(([1e-06], np.sort(modes_ev), [transport_energy_max_ev]))
    if np.any(np.diff(boundaries) <= 0.0):
        raise ValueError('mode energies must be distinct and strictly inside the transport interval')
    node_blocks, weight_blocks = ([], [])
    segment_nodes, segment_weights = leggauss(transport_order_per_segment)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        half = 0.5 * (upper - lower)
        node_blocks.append(lower + half * (segment_nodes + 1.0))
        weight_blocks.append(half * segment_weights)
    energy_ev = np.concatenate(node_blocks)
    energy_weights_ev = np.concatenate(weight_blocks)

    def _wave_vector(energy):
        return np.sqrt(2.0 * mass_ratio * _M_E * energy * _E_CHARGE * (1.0 + alpha_ev_inv * energy)) / _HBAR
    wave_vector = _wave_vector(energy_ev)
    elastic_q = 2.0 * wave_vector[:, None] * np.sin(scattering_angle[None, :] / 2.0)
    absorption_k = _wave_vector(energy_ev[None, :] + modes_ev[:, None])
    emission_k = _wave_vector(np.maximum(energy_ev[None, :] - modes_ev[:, None], 0.0))
    cosine = np.cos(scattering_angle)[None, None, :]
    initial = wave_vector[None, :, None]
    absorption_q = np.sqrt(initial ** 2 + absorption_k[:, :, None] ** 2 - 2.0 * initial * absorption_k[:, :, None] * cosine)
    emission_q = np.sqrt(initial ** 2 + emission_k[:, :, None] ** 2 - 2.0 * initial * emission_k[:, :, None] * cosine)
    inelastic_q = np.stack((absorption_q, emission_q), axis=0)
    thickness_m = thickness_angstrom * 1e-10
    rate_cache = {}

    def _rates(temperature, density_cm2):
        key = (float(temperature), float(density_cm2))
        if key in rate_cache:
            return rate_cache[key]
        chemical_potential, k_fermi = solve_kane_state_stage(float(temperature), float(density_cm2), mass_ratio, alpha_ev_inv, degeneracy, transport_energy_max_ev, density_order)
        if not np.isfinite(k_fermi) or k_fermi <= 0.0:
            raise ValueError('the sheet state requires a positive Fermi wave vector')
        elastic_pi = evaluate_polarizability_stage(elastic_q / k_fermi, k_fermi, float(temperature), chemical_potential, mass_ratio, alpha_ev_inv, degeneracy, polarizability_energy_max_ev, polarizability_order)
        elastic_screen = screen_dielectric_kernel_stage(elastic_q, elastic_pi, thickness_angstrom, epsilon_layer, epsilon_static)
        acoustic = compute_acoustic_rates_stage(energy_ev, scattering_angle, angle_weights, angle_weights, band_shift, elastic_screen[1], float(temperature), mass_ratio, alpha_ev_inv, c11_n_m, c12_n_m)
        impurity_coefficient = integrate_impurity_rate_coefficient_stage(energy_ev, scattering_angle, angle_weights, elastic_screen[0], mass_ratio, alpha_ev_inv)
        inelastic_pi = evaluate_polarizability_stage(inelastic_q / k_fermi, k_fermi, float(temperature), chemical_potential, mass_ratio, alpha_ev_inv, degeneracy, polarizability_energy_max_ev, polarizability_order)
        inelastic_eps = screen_dielectric_kernel_stage(inelastic_q, inelastic_pi, thickness_angstrom, epsilon_layer, epsilon_static)[1]
        surface = compute_surface_optical_rates_stage(energy_ev, scattering_angle, angle_weights, modes_mev, absorption_q, emission_q, absorption_k / wave_vector[None, :], emission_k / wave_vector[None, :], inelastic_eps[0], inelastic_eps[1], float(temperature), mass_ratio, thickness_angstrom, epsilon_static, epsilon_high_frequency, interfaces)
        rate_cache[key] = (float(chemical_potential), acoustic, impurity_coefficient, surface)
        return rate_cache[key]

    def _residual(log_state):
        density_cm2, impurity_cm2 = np.exp(log_state)
        residual = []
        for temperature, resistance, slope in zip(temperatures, resistances, slopes):
            chemical_potential, acoustic, impurity_coefficient, surface = _rates(temperature, density_cm2)
            response = compute_transport_moments_stage(energy_ev, energy_weights_ev, chemical_potential, acoustic, impurity_coefficient * impurity_cm2, surface, float(temperature), mass_ratio, alpha_ev_inv, degeneracy, thickness_angstrom)
            sheet_conductance = response[0] * thickness_m
            model_resistance = 1.0 / sheet_conductance
            model_slope = response[3] * 1e-06 / sheet_conductance ** 2
            residual.append(np.log(model_resistance / resistance))
            residual.append(np.log(model_slope / slope))
        return np.asarray(residual, dtype=float)
    lower = np.log(bounds[:, 0])
    upper = np.log(bounds[:, 1])
    solution = least_squares(_residual, 0.5 * (lower + upper), bounds=(lower, upper), method='trf', jac='3-point', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise ValueError('the sheet-state least-squares solve did not converge')
    return np.exp(solution.x).astype(float)

"""Transfer the inferred sheet to each encapsulation and return the defect ratio of the best one."""
import numpy as np
from numpy.polynomial.legendre import leggauss
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def select_encapsulation_defect_ratio(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), calibration_candidate_index: 'int'=0, target_temperature_k: 'float'=300.0, candidate_epsilon_static: 'Sequence[float]'=(3.9, 9.14, 12.53, 23.0), candidate_epsilon_high: 'Sequence[float]'=(2.5, 4.8, 3.2, 5.03), candidate_mode_energy_mev: 'Sequence[Sequence[float]]'=((55.6, 138.1), (81.4, 88.5), (48.18, 71.41), (12.4, 48.35)), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> float:
    """Reference composition of the inference stage and all transport stages."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _M_E = 9.1093837015e-31
    infer_sheet_state_stage = infer_sheet_state
    solve_kane_state_stage = solve_kane_state
    resolve_acoustic_tensor_stage = resolve_acoustic_tensor
    evaluate_polarizability_stage = evaluate_polarizability
    screen_dielectric_kernel_stage = screen_dielectric_kernel
    compute_acoustic_rates_stage = compute_acoustic_rates
    integrate_impurity_rate_coefficient_stage = integrate_impurity_rate_coefficient
    compute_surface_optical_rates_stage = compute_surface_optical_rates
    compute_transport_moments_stage = compute_transport_moments
    tensor_pairs = []
    for value, name in ((screened_dp_tensors_ev, 'screened_dp_tensors_ev'), (unscreened_dp_tensors_ev, 'unscreened_dp_tensors_ev')):
        try:
            pair = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(name + ' must be a (2, 3) array of finite numbers') from error
        if pair.shape != (2, 3) or not np.all(np.isfinite(pair)):
            raise ValueError(name + ' must be a (2, 3) array of finite numbers')
        tensor_pairs.append(pair)
    screened_pair, unscreened_pair = tensor_pairs
    index = calibration_candidate_index
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError('calibration_candidate_index must be an integer')
    target = float(target_temperature_k)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError('target_temperature_k must be positive and finite')
    static = np.asarray(candidate_epsilon_static, dtype=float)
    high = np.asarray(candidate_epsilon_high, dtype=float)
    try:
        modes = np.asarray(candidate_mode_energy_mev, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('candidate_mode_energy_mev must be a rectangular numeric table') from error
    if static.ndim != 1 or static.size < 2 or high.shape != static.shape:
        raise ValueError('at least two candidates with matching permittivity vectors are required')
    if modes.ndim != 2 or modes.shape[0] != static.size or modes.shape[1] == 0:
        raise ValueError('candidate_mode_energy_mev needs one nonempty row per candidate')
    if not (np.all(np.isfinite(static)) and np.all(np.isfinite(high)) and np.all(np.isfinite(modes))):
        raise ValueError('candidate tables must be finite')
    if np.any(high <= 0.0) or np.any(static < high) or np.any(modes <= 0.0):
        raise ValueError('candidates need static >= high-frequency > 0 and positive mode energies')
    if not 0 <= int(index) < static.size:
        raise ValueError('calibration_candidate_index is outside the candidate table')
    index = int(index)
    try:
        energy_max = float(transport_energy_max_ev)
    except (TypeError, ValueError) as error:
        raise ValueError('transport_energy_max_ev must be a number') from error
    for row_mev in modes:
        edges = np.concatenate(([1e-06], np.sort(row_mev * 0.001), [energy_max]))
        if not np.isfinite(energy_max) or np.any(np.diff(edges) <= 0.0):
            raise ValueError('each candidate needs distinct mode energies above 1e-3 meV and below the transport maximum')
    shared = dict(mass_ratio=mass_ratio, alpha_ev_inv=alpha_ev_inv, degeneracy=degeneracy, c11_n_m=c11_n_m, c12_n_m=c12_n_m, thickness_angstrom=thickness_angstrom, epsilon_layer=epsilon_layer, density_order=density_order, transport_order_per_segment=transport_order_per_segment, angle_order=angle_order, polarizability_order=polarizability_order, transport_energy_max_ev=transport_energy_max_ev, polarizability_energy_max_ev=polarizability_energy_max_ev, interface_count=interface_count)
    density_cm2, impurity_cm2 = infer_sheet_state_stage(screened_pair.tolist(), unscreened_pair.tolist(), temperatures_k=temperatures_k, sheet_resistance_ohm=sheet_resistance_ohm, hall_slope_ohm_per_tesla=hall_slope_ohm_per_tesla, epsilon_static=float(static[index]), epsilon_high_frequency=float(high[index]), mode_energy_mev=tuple((float(value) for value in modes[index])), density_bounds_cm2=density_bounds_cm2, impurity_bounds_cm2=impurity_bounds_cm2, **shared)
    angle_nodes, angle_weights = leggauss(int(angle_order))
    scattering_angle = np.pi * (angle_nodes + 1.0)
    angle_weights = np.pi * angle_weights
    order = int(angle_order)
    phonon_angle = scattering_angle[None, :] + 0.5 * scattering_angle[:, None] + 0.5 * np.pi
    band_shift = np.stack((resolve_acoustic_tensor_stage(phonon_angle, screened_pair[0])[0], resolve_acoustic_tensor_stage(phonon_angle, unscreened_pair[0])[0], resolve_acoustic_tensor_stage(phonon_angle, screened_pair[1])[1], resolve_acoustic_tensor_stage(phonon_angle, unscreened_pair[1])[1])).reshape(4, order, order)
    chemical_potential, k_fermi = solve_kane_state_stage(target, float(density_cm2), mass_ratio, alpha_ev_inv, degeneracy, transport_energy_max_ev, int(density_order))
    if not np.isfinite(k_fermi) or k_fermi <= 0.0:
        raise ValueError('the target state requires a positive Fermi wave vector')
    segment_nodes, segment_weights = leggauss(int(transport_order_per_segment))

    def _wave_vector(energy):
        return np.sqrt(2.0 * mass_ratio * _M_E * energy * _E_CHARGE * (1.0 + alpha_ev_inv * energy)) / _HBAR
    power_factors = []
    channel_grids = []
    for static_value, high_value, row_mev in zip(static, high, modes):
        modes_ev = row_mev * 0.001
        boundaries = np.concatenate(([1e-06], np.sort(modes_ev), [energy_max]))
        energy_ev = np.concatenate([lower + 0.5 * (upper - lower) * (segment_nodes + 1.0) for lower, upper in zip(boundaries[:-1], boundaries[1:])])
        energy_weights_ev = np.concatenate([0.5 * (upper - lower) * segment_weights for lower, upper in zip(boundaries[:-1], boundaries[1:])])
        wave_vector = _wave_vector(energy_ev)
        elastic_q = 2.0 * wave_vector[:, None] * np.sin(scattering_angle[None, :] / 2.0)
        absorption_k = _wave_vector(energy_ev[None, :] + modes_ev[:, None])
        emission_k = _wave_vector(np.maximum(energy_ev[None, :] - modes_ev[:, None], 0.0))
        cosine = np.cos(scattering_angle)[None, None, :]
        initial = wave_vector[None, :, None]
        absorption_q = np.sqrt(initial ** 2 + absorption_k[:, :, None] ** 2 - 2.0 * initial * absorption_k[:, :, None] * cosine)
        emission_q = np.sqrt(initial ** 2 + emission_k[:, :, None] ** 2 - 2.0 * initial * emission_k[:, :, None] * cosine)
        inelastic_q = np.stack((absorption_q, emission_q), axis=0)
        elastic_pi = evaluate_polarizability_stage(elastic_q / k_fermi, k_fermi, target, chemical_potential, mass_ratio, alpha_ev_inv, degeneracy, polarizability_energy_max_ev, int(polarizability_order))
        elastic_screen = screen_dielectric_kernel_stage(elastic_q, elastic_pi, thickness_angstrom, epsilon_layer, float(static_value))
        acoustic = compute_acoustic_rates_stage(energy_ev, scattering_angle, angle_weights, angle_weights, band_shift, elastic_screen[1], target, mass_ratio, alpha_ev_inv, c11_n_m, c12_n_m)
        impurity = integrate_impurity_rate_coefficient_stage(energy_ev, scattering_angle, angle_weights, elastic_screen[0], mass_ratio, alpha_ev_inv) * float(impurity_cm2)
        inelastic_pi = evaluate_polarizability_stage(inelastic_q / k_fermi, k_fermi, target, chemical_potential, mass_ratio, alpha_ev_inv, degeneracy, polarizability_energy_max_ev, int(polarizability_order))
        inelastic_eps = screen_dielectric_kernel_stage(inelastic_q, inelastic_pi, thickness_angstrom, epsilon_layer, float(static_value))[1]
        surface = compute_surface_optical_rates_stage(energy_ev, scattering_angle, angle_weights, row_mev, absorption_q, emission_q, absorption_k / wave_vector[None, :], emission_k / wave_vector[None, :], inelastic_eps[0], inelastic_eps[1], target, mass_ratio, thickness_angstrom, float(static_value), float(high_value), int(interface_count))
        response = compute_transport_moments_stage(energy_ev, energy_weights_ev, chemical_potential, acoustic, impurity, surface, target, mass_ratio, alpha_ev_inv, degeneracy, thickness_angstrom)
        power_factors.append(float(response[2]))
        channel_grids.append((energy_ev, energy_weights_ev, acoustic, impurity, surface))
    best = int(np.argmax(power_factors))
    energy_ev, energy_weights_ev, acoustic, impurity, surface = channel_grids[best]
    zero = np.zeros_like(energy_ev)
    phonon_limited = compute_transport_moments_stage(energy_ev, energy_weights_ev, chemical_potential, acoustic, zero, surface, target, mass_ratio, alpha_ev_inv, degeneracy, thickness_angstrom)
    impurity_limited = compute_transport_moments_stage(energy_ev, energy_weights_ev, chemical_potential, zero, impurity, zero, target, mass_ratio, alpha_ev_inv, degeneracy, thickness_angstrom)
    return float(phonon_limited[0] / impurity_limited[0])
SCICODE_GOLD_EOF
