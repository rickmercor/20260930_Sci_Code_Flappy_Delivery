"""
Infer the gate-held sheet density and midplane defect density from Hall-bar data.

The Hall bar sits between two identical dielectrics with static and high-frequency permittivities \(\kappa_0\) and \(\kappa_\infty\) and surface-mode energies \(\Omega_\nu\). Its sheet density \(n_s\) and midplane impurity density \(N_I\) are unknown and do not change with temperature. The arguments $screened_dp_tensors_ev$ and $unscreened_dp_tensors_ev$ are \((2,3)\) arrays with no defaults: in each, row 0 is a tensor \((\Xi_{xx},\Xi_{yy},\Xi_{xy})\) in eV whose step-02 LA shift is used, and row 1 is a tensor whose step-02 TA shift is used; the screened array supplies the shifts that step 05 divides by the dielectric response, and the unscreened array supplies the shifts that it does not. The scattering angles \(\phi_i\) and the incident directions \(\psi_j\) both use the Gauss-Legendre rule of order $angle_order$ on \([0,2\pi]\), and step 02 is evaluated at \(\theta_{ij}=\psi_j+\phi_i/2+\pi/2\) for every node pair, with the same weights serving as step 05's angle and incidence weights. For a trial pair, evaluate steps 01 through 08 at each tabulated temperature on the stack's own transport grid (Gauss-Legendre nodes of the given order on each open subinterval of \([10^{-6},E_{max}]\) eV split at the sorted mode energies), using \(\epsilon_e=\kappa_0\) and the impurity rate \(N_I\,\Gamma_I(1\ \mathrm{cm}^{-2})\). The model sheet conductance is \(\sigma_\square=\sigma a\), and the resistivity tensor is the inverse of the sheet conductivity tensor, so to first order in \(B\) the model sheet resistance is \(\rho_\square=1/\sigma_\square\) and the model weak-field Hall slope is



\[

\left|\frac{d\rho_{xy}}{dB}\right|=\frac{|\sigma_{xy}|/B}{\sigma_\square^{2}} .

\]



Minimize



\[

\chi^2(n_s,N_I)=\sum_j\left[\ln\frac{\rho_{\square,j}^{\,model}}{\rho_{\square,j}}\right]^2+\left[\ln\frac{|d\rho_{xy}/dB|_j^{\,model}}{|d\rho_{xy}/dB|_j}\right]^2

\]



over \(x=(\ln n_s,\ln N_I)\) inside the given bounds with `scipy.optimize.least_squares(residual, x0, bounds=(lower, upper), method="trf", jac="3-point", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)`. The residual vector lists, temperature by temperature in the given order, the resistance term and then the Hall term, and \(x_0\) is the midpoint of the logarithmic bounds. Temperatures must be distinct, positive, and finite, the energy maxima finite, and each tensor-pair argument a \((2,3)\) array of finite numbers, or `ValueError` is raised. Raise `ValueError` if the solver does not report success; errors raised by steps 01 through 08 or by the least-squares solver propagate. Implementations of this call reproduce the fitted pair only to about \(10^{-9}\) relative, so the tests compare \(\ln n_s\) and \(\ln N_I\) (densities in cm\(^{-2}\)) rounded to six decimals.

Returns
-------
A two-entry NumPy array containing \(n_s\) and \(N_I\), both in cm\(^{-2}\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_sheet_state(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), epsilon_static: 'float'=3.9, epsilon_high_frequency: 'float'=2.5, mode_energy_mev: 'Sequence[float]'=(55.6, 138.1), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> 'np.ndarray':
    """Return the least-squares sheet density and charged-defect density.

    The two tensor-pair arguments fix the acoustic stage. Row 0 of each pair
    is an in-plane tensor (Xi_xx, Xi_yy, Xi_xy) in eV that is contracted by
    the acoustic-tensor stage and whose LA row is used; row 1 is a tensor
    whose TA row is used. The screened pair supplies the shifts that are
    divided by the dielectric response and the unscreened pair supplies the
    shifts that are not. Both scattering angles phi_i and incident
    directions psi_j use the Gauss-Legendre rule of order angle_order on
    [0, 2 pi], and the tensors are contracted at the phonon directions
    theta_ij = psi_j + phi_i/2 + pi/2, measured counterclockwise from the
    tensor x axis.

    Parameters
    ----------
    screened_dp_tensors_ev, unscreened_dp_tensors_ev : array_like
        Shape (2, 3) arrays of finite tensor components in eV: row 0 for the
        LA shift and row 1 for the TA shift, as described above. These
        arguments have no defaults.
    temperatures_k : sequence of float
        At least two distinct positive temperatures in kelvin.
    sheet_resistance_ohm, hall_slope_ohm_per_tesla : sequence of float
        Positive sheet resistance in ohm and positive magnitude of the
        weak-field Hall slope d(rho_xy)/dB in ohm T^-1, one value per
        temperature and in the same order.
    epsilon_static, epsilon_high_frequency : float
        Static and high-frequency permittivities of the identical dielectric
        on both sides of the sheet, with epsilon_static >= epsilon_high_frequency > 0.
    mode_energy_mev : sequence of float
        Distinct surface-optical mode energies of that dielectric in meV,
        each above 1e-3 meV (the lower end of the transport grid) and below
        transport_energy_max_ev.
    density_bounds_cm2, impurity_bounds_cm2 : pair of float
        Finite search intervals (lower, upper) with 0 < lower < upper for the
        sheet density and the midplane impurity density, both in cm^-2.
    mass_ratio, alpha_ev_inv, degeneracy : float
        Kane band parameters: positive mass ratio, nonnegative
        nonparabolicity in eV^-1, positive degeneracy.
    c11_n_m, c12_n_m : float
        Two-dimensional elastic constants in N m^-1 with c11 > c12 >= 0.
    thickness_angstrom, epsilon_layer : float
        Positive layer thickness in angstrom and layer permittivity.
    density_order, transport_order_per_segment, angle_order, polarizability_order : int
        Positive integer quadrature orders (booleans are rejected).
    transport_energy_max_ev, polarizability_energy_max_ev : float
        Upper energies in eV of the transport grid (above 1e-6 eV) and of
        the thermal polarizability average.
    interface_count : int
        Positive integer number of identical dielectric interfaces.

    Returns
    -------
    numpy.ndarray
        Two entries: sheet electron density and midplane charged-impurity
        density, both in cm^-2.

    Notes
    -----
    The fit uses scipy.optimize.least_squares with method="trf",
    jac="3-point", xtol=ftol=gtol=1e-12, max_nfev=200, started at the
    midpoint of the logarithmic bounds. Implementations that follow this
    call agree only to about 1e-9 relative, because the solver's
    termination reacts to rounding in the residuals, so the tests compare
    the natural logarithms of the two densities rounded to six decimals.

    Raises
    ------
    ValueError
        For fewer than two temperatures, repeated, nonpositive, or nonfinite
        temperatures, or data that are not positive, finite, and matched one
        to one with the temperatures; mode energies that are empty, not positive, not
        distinct, at or below 1e-3 meV, or at or above
        transport_energy_max_ev; search intervals that are not finite pairs
        with 0 < lower < upper; a tensor-pair argument that is not a (2, 3)
        array of finite numbers; nonfinite material parameters or energy
        maxima, permittivities violating
        epsilon_static >= epsilon_high_frequency > 0, nonpositive mass,
        degeneracy, thickness, or layer permittivity, a negative
        alpha_ev_inv, or elastic constants violating c11 > c12 >= 0;
        nonpositive energy maxima or a transport maximum at
        or below 1e-6 eV; orders or interface_count that are not positive
        integers (booleans included); a trial state whose Fermi wave vector
        is zero at a tabulated temperature (chemical potential at or below
        the band edge); a least-squares solve that does not report
        convergence; or any error raised by stages 01 through 08 on the
        trial grids or by the least-squares solver (for example a negative
        summed surface-optical rate when angle_order is 2).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Infer the gate-held sheet density and midplane defect density from Hall-bar data."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def _oracle_infer_sheet_state(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), epsilon_static: 'float'=3.9, epsilon_high_frequency: 'float'=2.5, mode_energy_mev: 'Sequence[float]'=(55.6, 138.1), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> 'np.ndarray':
    """Reference least-squares inference through the full transport chain."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _M_E = 9.1093837015e-31
    solve_kane_state_stage = _oracle_solve_kane_state
    resolve_acoustic_tensor_stage = _oracle_resolve_acoustic_tensor
    evaluate_polarizability_stage = _oracle_evaluate_polarizability
    screen_dielectric_kernel_stage = _oracle_screen_dielectric_kernel
    compute_acoustic_rates_stage = _oracle_compute_acoustic_rates
    integrate_impurity_rate_coefficient_stage = _oracle_integrate_impurity_rate_coefficient
    compute_surface_optical_rates_stage = _oracle_compute_surface_optical_rates
    compute_transport_moments_stage = _oracle_compute_transport_moments

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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "orders = dict(density_order=60, transport_order_per_segment=8, angle_order=12, polarizability_order=24); params = dict(screened_dp_tensors_ev=((5.2, 4.1, 0.6), (0.9, -0.4, 0.3)), unscreened_dp_tensors_ev=((1.1, -0.7, 0.25), (6.3, 4.9, -0.45)), temperatures_k=(160.0, 230.0, 290.0), sheet_resistance_ohm=(910.7, 1555.0, 2418.0), hall_slope_ohm_per_tesla=(47.55, 52.07, 54.6), **orders)",
            "call": "np.round(np.log(infer_sheet_state(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_infer_sheet_state(**params)), 6)",
        },
        {
            "setup": "orders = dict(density_order=60, transport_order_per_segment=8, angle_order=12, polarizability_order=24); params = dict(screened_dp_tensors_ev=((4.6, 5.5, -0.35), (1.2, 2.0, 0.15)), unscreened_dp_tensors_ev=((-0.8, 1.2, 0.4), (5.1, 6.6, 0.7)), temperatures_k=(200.0, 280.0), sheet_resistance_ohm=(1211.0, 2131.0), hall_slope_ohm_per_tesla=(42.17, 46.78), epsilon_static=9.14, epsilon_high_frequency=4.8, mode_energy_mev=(81.4, 88.5), density_bounds_cm2=(1.0e13, 2.5e13), impurity_bounds_cm2=(3.0e11, 6.0e12), **orders)",
            "call": "np.round(np.log(infer_sheet_state(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_infer_sheet_state(**params)), 6)",
        },
        {
            "setup": "orders = dict(density_order=48, transport_order_per_segment=6, angle_order=10, polarizability_order=20); params = dict(screened_dp_tensors_ev=((6.9, 3.2, 0.8), (1.4, 0.5, -0.6)), unscreened_dp_tensors_ev=((0.3, 2.2, -0.15), (7.4, 3.8, 0.95)), temperatures_k=(150.0, 200.0, 250.0, 310.0), sheet_resistance_ohm=(6676.0, 8877.0, 12150.0, 16900.0), hall_slope_ohm_per_tesla=(26.64, 28.86, 31.48, 33.05), epsilon_static=23.0, epsilon_high_frequency=5.03, mode_energy_mev=(12.4, 48.35), density_bounds_cm2=(1.0e13, 3.0e13), impurity_bounds_cm2=(1.0e12, 1.5e13), mass_ratio=0.45, alpha_ev_inv=0.5, degeneracy=4.0, **orders)",
            "call": "np.round(np.log(infer_sheet_state(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_infer_sheet_state(**params)), 6)",
        },
        {
            "setup": "fast = dict(density_order=20, transport_order_per_segment=4, angle_order=6, polarizability_order=10, screened_dp_tensors_ev=((5.2, 4.1, 0.6), (0.9, -0.4, 0.3)), unscreened_dp_tensors_ev=((1.1, -0.7, 0.25), (6.3, 4.9, -0.45)))\npairs = dict(screened_dp_tensors_ev=((5.2, 4.1, 0.6), (0.9, -0.4, 0.3)), unscreened_dp_tensors_ev=((1.1, -0.7, 0.25), (6.3, 4.9, -0.45)))\ndef bad_keyword_sets():\n    return [\n        dict(fast, temperatures_k=(300.0,), sheet_resistance_ohm=(1000.0,), hall_slope_ohm_per_tesla=(50.0,)),\n        dict(fast, sheet_resistance_ohm=(753.6, -1334.0, 2221.0)),\n        dict(fast, hall_slope_ohm_per_tesla=(66.16, np.nan, 74.19)),\n        dict(fast, density_bounds_cm2=(3.0e13, 1.0e13)),\n        dict(pairs, density_order=True, transport_order_per_segment=4, angle_order=6, polarizability_order=10),\n        dict(fast, epsilon_static=2.0, epsilon_high_frequency=2.5),\n        dict(fast, temperatures_k=(200.0, 200.0, 300.0)),\n        dict(fast, mode_energy_mev=(55.60, 55.60)),\n        dict(fast, mode_energy_mev=(5.0e-4, 138.10)),\n        dict(fast, density_bounds_cm2=(1.0e10, 2.0e10)),\n        dict(fast, temperatures_k=(200.0, 250.0)),\n        dict(fast, mode_energy_mev=(55.60, 900.0)),\n        dict(fast, mode_energy_mev=()),\n        dict(fast, mode_energy_mev=(-55.60, 138.10)),\n        dict(fast, impurity_bounds_cm2=(1.0e11, np.nan)),\n        dict(fast, c11_n_m=np.nan),\n        dict(fast, mass_ratio=0.0),\n        dict(fast, screened_dp_tensors_ev=((5.2, 4.1), (0.9, -0.4))),\n        dict(fast, c12_n_m=140.0),\n        dict(fast, transport_energy_max_ev=1.0e-7),\n        dict(fast, interface_count=0),\n        dict(pairs, density_order=20, transport_order_per_segment=4, angle_order=6.5, polarizability_order=10),\n        dict(pairs, density_order=10, transport_order_per_segment=3, angle_order=4, polarizability_order=7, hall_slope_ohm_per_tesla=(76.39, 71.3, 76.24), sheet_resistance_ohm=(118.1, 133.9, 1194.0), epsilon_static=23.0, epsilon_high_frequency=5.03, mode_energy_mev=(12.4, 48.35)),\n        dict(fast, polarizability_energy_max_ev=0.0),\n        dict(pairs, density_order=20, transport_order_per_segment=4, angle_order=2, polarizability_order=10),\n        dict(fast, hall_slope_ohm_per_tesla=(66.16, 0.0, 74.19)),\n        dict(fast, sheet_resistance_ohm=(753.6, np.inf, 2221.0)),\n        dict(fast, density_bounds_cm2=(1.0e13, 2.0e13, 3.0e13), impurity_bounds_cm2=(1.0e11, 5.0e11, 1.0e13)),\n        dict(fast, impurity_bounds_cm2=(0.0, 1.0e13)),\n        dict(fast, epsilon_static=np.nan),\n        dict(fast, thickness_angstrom=np.inf),\n        dict(fast, epsilon_static=3.90, epsilon_high_frequency=0.0),\n        dict(fast, degeneracy=0.0),\n        dict(fast, thickness_angstrom=0.0),\n        dict(fast, epsilon_layer=-7.6),\n        dict(fast, unscreened_dp_tensors_ev=((1.1, np.nan, 0.25), (6.3, 4.9, -0.45))),\n        dict(fast, alpha_ev_inv=-0.1),\n        dict(fast, c12_n_m=-1.0),\n        dict(pairs, density_order=20, transport_order_per_segment=0, angle_order=6, polarizability_order=10),\n        dict(pairs, density_order=20, transport_order_per_segment=4, angle_order=6, polarizability_order=-10),\n        dict(fast, temperatures_k=(200.0, -250.0, 300.0)),\n        dict(fast, temperatures_k=(200.0, np.nan, 300.0)),\n        dict(fast, transport_energy_max_ev=np.inf),\n        dict(fast, polarizability_energy_max_ev=np.inf),\n        dict(fast, screened_dp_tensors_ev=((5.2, 4.1, np.inf), (0.9, -0.4, 0.3))),\n        dict(fast, unscreened_dp_tensors_ev=(1.1, -0.7, 0.25)),\n        dict(fast, screened_dp_tensors_ev='5.2 4.1 0.6 0.9 -0.4 0.3'),\n        dict(fast, unscreened_dp_tensors_ev=((1.1, -0.7, 0.25), (6.3, 4.9))),\n        dict(fast, screened_dp_tensors_ev=((5.2, 4.1, 0.6), (0.9, -0.4, 0.3), (0.0, 0.0, 0.0))),\n    ]\ndef rejected_public():\n    count = 0.0\n    for kwargs in bad_keyword_sets():\n        try:\n            infer_sheet_state(**kwargs)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for kwargs in bad_keyword_sets():\n        try:\n            _oracle_infer_sheet_state(**kwargs)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
