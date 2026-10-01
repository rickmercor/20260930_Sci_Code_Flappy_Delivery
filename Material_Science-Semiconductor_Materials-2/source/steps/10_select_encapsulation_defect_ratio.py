"""
Transfer the inferred sheet to each encapsulation and return the defect ratio of the best one.

Apply step 09 with the candidate row selected by $calibration_candidate_index$ to obtain \(n_s\) and \(N_I\). For every candidate row in order, place the same sheet, with the same \(n_s\) and \(N_I\), between two layers of that candidate and evaluate steps 01 through 08 at $target_temperature_k$ on that candidate's own transport grid, built as in step 09 with Gauss-Legendre nodes, with the same tensor-pair arguments, angle and incidence nodes, and phonon directions as step 09. Use the candidate's \(\kappa_0\) as \(\epsilon_e\) in the image-charge and screening stages and its \(\kappa_0\), \(\kappa_\infty\), and \(\Omega_\nu\) in the surface-optical stage. Select the first candidate in table order with the largest \(P_i=10^3S_i^2\sigma_i\). For that candidate, apply step 08 twice more on its grid: once with the acoustic and surface-optical rates and a zero impurity rate, giving \(\sigma_{ph}\), and once with the impurity rate at \(N_I\) and zero acoustic and surface-optical rates, giving \(\sigma_I\). Because \(\sigma_I\propto1/N_I\), the critical density at which the impurity-only conductivity would equal \(\sigma_{ph}\) is \(N_{cr}=N_I\sigma_I/\sigma_{ph}\). A tensor-pair argument that is not a \((2,3)\) array of finite numbers, empty mode rows, a $transport_energy_max_ev$ that is not a finite number, and a zero Fermi wave vector at $target_temperature_k$ raise `ValueError`, and errors raised by step 09 or by steps 01 through 08 at the target temperature propagate. Return



\[

\frac{N_I}{N_{cr}}=\frac{\sigma_{ph}}{\sigma_I}.

\]

Returns
-------
A native Python float containing \(N_I/N_{cr}\), dimensionless. The tests compare its natural logarithm rounded to six decimals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_encapsulation_defect_ratio(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), calibration_candidate_index: 'int'=0, target_temperature_k: 'float'=300.0, candidate_epsilon_static: 'Sequence[float]'=(3.9, 9.14, 12.53, 23.0), candidate_epsilon_high: 'Sequence[float]'=(2.5, 4.8, 3.2, 5.03), candidate_mode_energy_mev: 'Sequence[Sequence[float]]'=((55.6, 138.1), (81.4, 88.5), (48.18, 71.41), (12.4, 48.35)), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> float:
    """Return N_I / N_cr for the candidate stack with the largest power factor.

    Parameters
    ----------
    screened_dp_tensors_ev, unscreened_dp_tensors_ev : array_like
        Shape (2, 3) tensor pairs in eV with the layout and meaning used by
        the inference stage (row 0 for the LA shift, row 1 for the TA shift),
        passed unchanged to the inference stage and used in the same way,
        with the same Gauss-Legendre scattering and incidence nodes and the
        same phonon directions theta_ij = psi_j + phi_i/2 + pi/2, at the
        target temperature. These arguments have no defaults.
    temperatures_k, sheet_resistance_ohm, hall_slope_ohm_per_tesla : sequence of float
        Hall-bar data of the sheet in its calibration stack, in the layout
        required by the sheet-state inference stage.
    calibration_candidate_index : int
        Zero-based row of the candidate tables that describes the stack in
        which the Hall-bar data were taken (booleans are rejected).
    target_temperature_k : float
        Positive temperature in kelvin at which every candidate is compared.
    candidate_epsilon_static, candidate_epsilon_high : sequence of float
        One static and one high-frequency permittivity per candidate stack,
        with static >= high-frequency > 0; at least two candidates.
    candidate_mode_energy_mev : sequence of sequence of float
        Rectangular table with one row of distinct surface-optical mode
        energies in meV per candidate, each above 1e-3 meV and below
        transport_energy_max_ev.
    density_bounds_cm2, impurity_bounds_cm2 : pair of float
        Search intervals passed unchanged to the inference stage.
    mass_ratio, alpha_ev_inv, degeneracy, c11_n_m, c12_n_m,
    thickness_angstrom, epsilon_layer, density_order, transport_order_per_segment,
    angle_order, polarizability_order, transport_energy_max_ev,
    polarizability_energy_max_ev, interface_count :
        Material, quadrature, and interface parameters with the same meaning
        and domains as in the inference stage; the same values and the same
        grid construction are used for the calibration and for every
        candidate.

    Returns
    -------
    float
        With the inferred sheet density n_s and charged-impurity density N_I
        unchanged, every candidate is evaluated at target_temperature_k and
        the first candidate in table order with the largest
        P = 10^3 S^2 sigma is selected. For that candidate, sigma_ph is the
        conductivity with the acoustic and surface-optical rates only and
        sigma_I is the conductivity with the impurity rate at N_I only; the
        returned value is N_I / N_cr = sigma_ph / sigma_I, where N_cr is the
        impurity density at which the impurity-only conductivity equals
        sigma_ph.

    Notes
    -----
    The tests compare the natural logarithm of the returned value rounded
    to six decimals, because the inferred sheet state is reproducible only
    to about 1e-9 relative.

    Raises
    ------
    ValueError
        For a tensor-pair argument that is not a (2, 3) array of finite
        numbers; a calibration index that is not an integer (booleans
        included) or lies outside the candidate table; a nonpositive or
        nonfinite target temperature; fewer than two candidates, mismatched
        or ragged candidate tables, empty mode rows, or nonfinite entries;
        permittivities violating static >= high-frequency > 0; mode energies
        that are not positive, not distinct within a row, at or below
        1e-3 meV, or at or above transport_energy_max_ev, or a
        transport_energy_max_ev that is not a finite number; a zero Fermi
        wave vector of the inferred sheet at target_temperature_k (chemical
        potential at or below the band edge); or any error raised by the
        inference stage or by stages 01 through 08 at target_temperature_k
        (for example a zero conductivity moment when the target is so cold
        that the thermal window underflows at every transport node).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Transfer the inferred sheet to each encapsulation and return the defect ratio of the best one."""
import numpy as np
from numpy.polynomial.legendre import leggauss
_E_CHARGE = 1.602176634e-19
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31

def _oracle_select_encapsulation_defect_ratio(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), calibration_candidate_index: 'int'=0, target_temperature_k: 'float'=300.0, candidate_epsilon_static: 'Sequence[float]'=(3.9, 9.14, 12.53, 23.0), candidate_epsilon_high: 'Sequence[float]'=(2.5, 4.8, 3.2, 5.03), candidate_mode_energy_mev: 'Sequence[Sequence[float]]'=((55.6, 138.1), (81.4, 88.5), (48.18, 71.41), (12.4, 48.35)), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> float:
    """Reference composition of the inference stage and all transport stages."""
    _E_CHARGE = 1.602176634e-19
    _HBAR = 1.054571817e-34
    _M_E = 9.1093837015e-31
    infer_sheet_state_stage = _oracle_infer_sheet_state
    solve_kane_state_stage = _oracle_solve_kane_state
    resolve_acoustic_tensor_stage = _oracle_resolve_acoustic_tensor
    evaluate_polarizability_stage = _oracle_evaluate_polarizability
    screen_dielectric_kernel_stage = _oracle_screen_dielectric_kernel
    compute_acoustic_rates_stage = _oracle_compute_acoustic_rates
    integrate_impurity_rate_coefficient_stage = _oracle_integrate_impurity_rate_coefficient
    compute_surface_optical_rates_stage = _oracle_compute_surface_optical_rates
    compute_transport_moments_stage = _oracle_compute_transport_moments
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and rejection test specifications."""
    return [
        {
            "setup": "fast = dict(density_order=60, transport_order_per_segment=8, angle_order=12, polarizability_order=24); params = dict(screened_dp_tensors_ev=((5.7, 4.9, 0.45), (0.6, -1.1, 0.2)), unscreened_dp_tensors_ev=((1.9, -1.3, -0.55), (4.4, 6.1, 0.35)), temperatures_k=(150.0, 225.0, 300.0), sheet_resistance_ohm=(749.7, 1561.0, 3050.0), hall_slope_ohm_per_tesla=(60.95, 68.25, 73.1), **fast)",
            "call": "np.round(np.log(select_encapsulation_defect_ratio(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_select_encapsulation_defect_ratio(**params)), 6)",
        },
        {
            "setup": "fast = dict(density_order=60, transport_order_per_segment=8, angle_order=12, polarizability_order=24); params = dict(screened_dp_tensors_ev=((4.3, 6.4, -0.6), (1.7, -0.9, 0.45)), unscreened_dp_tensors_ev=((0.8, 0.3, 0.2), (3.9, 5.2, -0.75)), temperatures_k=(180.0, 260.0), sheet_resistance_ohm=(6224.0, 11140.0), hall_slope_ohm_per_tesla=(41.35, 42.86), calibration_candidate_index=1, target_temperature_k=240.0, candidate_epsilon_static=(3.9, 23.0), candidate_epsilon_high=(2.5, 5.03), candidate_mode_energy_mev=((55.6, 138.1), (12.4, 48.35)), density_bounds_cm2=(1.0e13, 3.0e13), impurity_bounds_cm2=(1.0e12, 2.0e13), **fast)",
            "call": "np.round(np.log(select_encapsulation_defect_ratio(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_select_encapsulation_defect_ratio(**params)), 6)",
        },
        {
            "setup": "fast = dict(density_order=48, transport_order_per_segment=6, angle_order=10, polarizability_order=20); params = dict(screened_dp_tensors_ev=((6.2, 5.6, 0.3), (0.5, 0.2, 1.1)), unscreened_dp_tensors_ev=((1.45, -0.6, 0.4), (7.65, 4.75, 0.8)), temperatures_k=(140.0, 200.0, 260.0, 320.0), sheet_resistance_ohm=(826.1, 1374.0, 2292.0, 3660.0), hall_slope_ohm_per_tesla=(52.23, 57.13, 61.58, 64.3), calibration_candidate_index=2, target_temperature_k=320.0, candidate_epsilon_static=(9.14, 12.53, 3.9), candidate_epsilon_high=(4.8, 3.2, 2.5), candidate_mode_energy_mev=((81.4, 88.5), (48.18, 71.41), (55.6, 138.1)), density_bounds_cm2=(1.2e13, 3.0e13), impurity_bounds_cm2=(1.0e11, 1.0e13), mass_ratio=0.42, alpha_ev_inv=0.6, **fast)",
            "call": "np.round(np.log(select_encapsulation_defect_ratio(**params)), 6)",
            "gold_call": "np.round(np.log(_oracle_select_encapsulation_defect_ratio(**params)), 6)",
        },
        {
            "setup": "fast = dict(density_order=20, transport_order_per_segment=4, angle_order=6, polarizability_order=10, screened_dp_tensors_ev=((5.7, 4.9, 0.45), (0.6, -1.1, 0.2)), unscreened_dp_tensors_ev=((1.9, -1.3, -0.55), (4.4, 6.1, 0.35)))\npairs = dict(screened_dp_tensors_ev=((5.7, 4.9, 0.45), (0.6, -1.1, 0.2)), unscreened_dp_tensors_ev=((1.9, -1.3, -0.55), (4.4, 6.1, 0.35)))\ndef bad_keyword_sets():\n    return [\n        dict(fast, calibration_candidate_index=True),\n        dict(fast, calibration_candidate_index=4),\n        dict(fast, target_temperature_k=-300.0),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40,), (48.18, 71.41), (12.40, 48.35))),\n        dict(fast, candidate_epsilon_static=(3.90, 9.14, 2.00, 23.00)),\n        dict(fast, candidate_epsilon_static=(3.90,), candidate_epsilon_high=(2.50,), candidate_mode_energy_mev=((55.60, 138.10),)),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, 81.40), (48.18, 71.41), (12.40, 48.35))),\n        dict(fast, target_temperature_k=1000.0),\n        dict(fast, target_temperature_k=np.nan),\n        dict(fast, candidate_epsilon_high=(2.50, 4.80, 3.20)),\n        dict(fast, candidate_epsilon_static=(3.90, np.nan, 12.53, 23.00)),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, 900.0), (48.18, 71.41), (12.40, 48.35))),\n        dict(fast, calibration_candidate_index=0.0),\n        dict(fast, calibration_candidate_index=-1),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, 88.50), (-48.18, 71.41), (12.40, 48.35))),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, 88.50), (5.0e-4, 71.41), (12.40, 48.35))),\n        dict(fast, density_bounds_cm2=(1.0e10, 2.0e10)),\n        dict(pairs, temperatures_k=(200.0, 250.0), sheet_resistance_ohm=(753.6, 1334.0), hall_slope_ohm_per_tesla=(66.16, 70.96), density_order=20, transport_order_per_segment=4, angle_order=2, polarizability_order=10),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, 88.50), (48.18, 71.41))),\n        dict(fast, candidate_epsilon_high=(2.50, np.inf, 3.20, 5.03)),\n        dict(fast, candidate_mode_energy_mev=((55.60, 138.10), (81.40, np.nan), (48.18, 71.41), (12.40, 48.35))),\n        dict(fast, candidate_epsilon_high=(2.50, 0.0, 3.20, 5.03)),\n        dict(fast, candidate_mode_energy_mev=((), (), (), ())),\n        dict(fast, transport_energy_max_ev=np.inf),\n        dict(fast, transport_energy_max_ev=None),\n        dict(fast, target_temperature_k=0.002),\n        dict(fast, screened_dp_tensors_ev=((5.7, np.nan, 0.45), (0.6, -1.1, 0.2))),\n        dict(fast, unscreened_dp_tensors_ev=((1.9, -1.3), (4.4, 6.1))),\n        dict(fast, screened_dp_tensors_ev='5.7 4.9 0.45 0.6 -1.1 0.2'),\n        dict(fast, unscreened_dp_tensors_ev=((1.9, -1.3, -0.55), (4.4, 6.1, np.inf))),\n        dict(fast, screened_dp_tensors_ev=(5.7, 4.9, 0.45)),\n        dict(fast, target_temperature_k=5000.0),\n    ]\ndef rejected_public():\n    count = 0.0\n    for kwargs in bad_keyword_sets():\n        try:\n            select_encapsulation_defect_ratio(**kwargs)\n        except ValueError:\n            count += 1.0\n    return count\ndef rejected_oracle():\n    count = 0.0\n    for kwargs in bad_keyword_sets():\n        try:\n            _oracle_select_encapsulation_defect_ratio(**kwargs)\n        except ValueError:\n            count += 1.0\n    return count",
            "call": "rejected_public()",
            "gold_call": "rejected_oracle()",
        },
    ]
