"""
Compose every preceding public function to select the task-defined admissible STEM design. Exact and linearized paths apply propagation only between retained gratings. For the selected row, compare the once-normalized full linearized exit with the exact exit over the complete voltage/defocus/hypothesis/position scan, using E_lin=max ||d_lin-d_exact||_2/||d_exact||_2. Define M=min_voltage[1000*lambda/(2*dx)-max(alpha,outer_edge)] mrad, theta0=1 mrad and mu=M/theta0. With f_D=scan_count*(2*slices-1)*(2^logical_qubits-2)/total_scan_CX, return round(total_scan_CX*E_ref*E_lin/(C_min*mu*f_D),6). R is a dimensionless task-authored post-selection audit, not a published QuScope quantity. Resolve the linearized wave with bivariate_multislice_orders using B=H0 and D=H1-H0, and obtain its detector probability from annular_order_coherence at (a,b)=(1,0) or (1,1), respectively. The full-space coherence channel supplies the normalization of the complete reconstructed exit. This coherent order representation is an exact re-expression of the existing linearized model, not an additional approximation or a new resource charge.

The source-based STEM propagation and resource boundary feed a task-defined convergence, discrimination and model-form audit.

Returns
-------
float, Finite dimensionless audit score rounded once to six decimal places.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quscope_quantum_resource_design(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> float:
    """Compose all eleven preceding public functions for the declared instance.

Parameters
----------
candidates : array_like
    Nonempty finite (C,5) array-like rows [alpha,e0,e1,e2,e3] in mrad, with alpha>0 and 0<=e0<e1<e2<e3. Each candidate must have nonempty detector masks and nonzero detector-vector norm on the reference grid.
voltages_kv : array_like
    Nonempty one-dimensional array-like positive finite accelerating voltages in kV.
defocuses_a : array_like
    Nonempty one-dimensional array-like finite defocus values in Å.
n_values : array_like
    Nonempty ordered array-like of power-of-two grid dimensions at least 2.
group_factors : array_like
    Nonempty ordered array-like grouping factors drawn from 1,2,3,6.
field_of_view_a : float
    Positive finite side length of the periodic square cell, in Å.
base_slice_thickness_a : float
    Positive finite thickness of one base slice, in Å.
cs_mm : float
    Finite nonnegative spherical-aberration coefficient, in mm.
reference_n : int
    Power-of-two dimension of the ungrouped reference, no smaller than every candidate grid.
detector_error_tolerance : float
    Positive finite ceiling on reference-relative detector error.
contrast_floor : float
    Finite nonnegative floor on same-position H1/H0 detector separation.
fidelity_floor : float
    Finite normalized quantum/classical fidelity floor in (0,1].
scan_positions_a : array_like
    Finite array-like shape (S,2), S>=2, of scan positions in Å, in the declared order.

Returns
-------
result : float
    Finite dimensionless audit score rounded once to six decimal places.

Notes
-----
Compose all eleven preceding public functions for the declared instance.

The linearized detector uses bivariate_multislice_orders and
annular_order_coherence: B is the H0 stack and D is H1-H0, with
(a,b)=(1,0) for H0 and (1,1) for H1. Use their coherent full-space
normalization, combining their outputs rather than reimplementing them.

Inputs have the same meanings and ordering as quantum_resource_design_table.
Returns one finite Python float: the detector-model/resource-distortion
dimensionless score rounded once to six decimal places, with angular
margin divided by the fixed physical scale 1 mrad.

Raises
------
ValueError
    If no candidate satisfies all feasibility constraints."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_quscope_quantum_resource_design(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> float:
    """Compose the robust QuScope design and strong-phase resource score."""
    constants = _oracle_relativistic_electron_constants(float(voltages_kv[0]))
    n0 = int(n_values[0])
    dx0 = float(field_of_view_a) / n0
    g0 = int(group_factors[0])
    stack = _oracle_build_grouped_strong_scattering_stack(n0, dx0, 0, g0)
    probe = _oracle_amplitude_encoded_stem_probe(n0, dx0, constants[0], float(candidates[0][0]), float(defocuses_a[0]), cs_mm, scan_positions_a[0])
    first_block = _oracle_quantum_slice_block(probe, stack[0], constants[0], constants[1], base_slice_thickness_a * g0, dx0, propagate_after=len(stack) > 1)
    exit_state = _oracle_quantum_multislice_exit_state(stack, probe, constants[0], constants[1], base_slice_thickness_a, g0, dx0)
    resources = _oracle_transpiled_multislice_resources(n0, 6 // g0)
    if first_block.shape != probe.shape or resources.shape != (4,):
        raise ValueError('invalid composed public-step output')
    table = _oracle_quantum_resource_design_table(candidates, voltages_kv, defocuses_a, n_values, group_factors, field_of_view_a, base_slice_thickness_a, cs_mm, reference_n, detector_error_tolerance, contrast_floor, fidelity_floor, scan_positions_a)
    selected = _oracle_select_quantum_resource_design(table)
    candidate_index = int(selected[1])
    n = int(selected[2])
    group = int(selected[3])
    slice_count = int(selected[4])
    logical_qubits = int(selected[5])
    dx = float(field_of_view_a) / n
    maximum_linearized_detector_error = -np.inf
    detector_edges = np.asarray(candidates, dtype=float)[candidate_index, 1:]
    positions = np.asarray(scan_positions_a, dtype=float)
    baseline_stack = _oracle_build_grouped_strong_scattering_stack(n, dx, 0, group)
    contrast_stack = _oracle_build_grouped_strong_scattering_stack(n, dx, 1, group) - baseline_stack
    order_pairs = [(degree - q, q) for degree in range(slice_count + 1) for q in range(degree + 1)]
    for voltage in np.asarray(voltages_kv, dtype=float):
        (wavelength, sigma) = _oracle_relativistic_electron_constants(float(voltage))
        for defocus in np.asarray(defocuses_a, dtype=float):
            for position in positions:
                incident = _oracle_amplitude_encoded_stem_probe(n, dx, wavelength, float(np.asarray(candidates)[candidate_index, 0]), float(defocus), cs_mm, position)
                orders = _oracle_bivariate_multislice_orders(baseline_stack, contrast_stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                coherence = _oracle_annular_order_coherence(orders, wavelength, dx, detector_edges)
                for hypothesis in (0, 1):
                    grouped = baseline_stack + hypothesis * contrast_stack
                    exact = _oracle_quantum_multislice_exit_state(grouped, incident, wavelength, sigma, base_slice_thickness_a, group, dx)
                    weights = np.asarray([float(hypothesis ** q) for (p, q) in order_pairs])
                    coherent_signal = np.einsum('a,cab,b->c', weights, coherence, weights).real
                    if coherent_signal[3] <= 0.0 or not np.all(np.isfinite(coherent_signal)):
                        raise ValueError('invalid reconstructed linearized wave norm')
                    linearized_detector = coherent_signal[:3] / coherent_signal[3]
                    exact_detector = _oracle_annular_detector_vector(exact, wavelength, dx, detector_edges)
                    mismatch = float(np.linalg.norm(linearized_detector - exact_detector) / np.linalg.norm(exact_detector))
                    maximum_linearized_detector_error = max(maximum_linearized_detector_error, mismatch)
    detector_outer = max(float(candidates[candidate_index][0]), float(candidates[candidate_index][-1]))
    nyquist_margin = min((1000.0 * _oracle_relativistic_electron_constants(float(voltage))[0] / (2.0 * dx) - detector_outer for voltage in np.asarray(voltages_kv, dtype=float)))
    scan_count = np.asarray(scan_positions_a, dtype=float).shape[0]
    diagonal_scan_cx = scan_count * (2 * slice_count - 1) * (2 ** logical_qubits - 2)
    diagonal_cx_fraction = diagonal_scan_cx / selected[10]
    angular_scale_mrad = 1.0
    dimensionless_margin = nyquist_margin / angular_scale_mrad
    score = selected[10] * selected[8] * maximum_linearized_detector_error / (selected[9] * dimensionless_margin * diagonal_cx_fraction)
    return float(round(score, 6))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'group_factors_gold = copy.deepcopy(group_factors)\n'
               'n_values_gold = copy.deepcopy(n_values)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates,voltages,defocuses,n_values,group_factors)',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold, voltages_gold, defocuses_gold, '
                   'n_values_gold, group_factors_gold)'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'group_factors_gold = copy.deepcopy(group_factors)\n'
               'n_values_gold = copy.deepcopy(n_values)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates,voltages,defocuses,n_values,group_factors,contrast_floor=0.125)',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold, voltages_gold, defocuses_gold, '
                   'n_values_gold, group_factors_gold, contrast_floor=0.125)'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'n_values_gold = copy.deepcopy(n_values)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates,voltages,defocuses,n_values,np.array([1]))',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold, voltages_gold, defocuses_gold, '
                   'n_values_gold, np.array([1]))'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'group_factors_gold = copy.deepcopy(group_factors)\n'
               'n_values_gold = copy.deepcopy(n_values)\n'
               'scan_positions_gold = copy.deepcopy(scan_positions)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates,voltages,defocuses,n_values,group_factors,scan_positions_a=scan_positions[[0,3]])',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold, voltages_gold, defocuses_gold, '
                   'n_values_gold, group_factors_gold, scan_positions_a=scan_positions_gold[[0, 3]])'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'scan_positions_gold = copy.deepcopy(scan_positions)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates[[1]],voltages,defocuses,np.array([32]),np.array([1]),scan_positions_a=scan_positions)',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold[[1]], voltages_gold, defocuses_gold, '
                   'np.array([32]), np.array([1]), scan_positions_a=scan_positions_gold)'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'group_factors_gold = copy.deepcopy(group_factors)\n'
               'n_values_gold = copy.deepcopy(n_values)\n'
               'scan_positions_gold = copy.deepcopy(scan_positions)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates,voltages,defocuses,n_values,group_factors,contrast_floor=0.122,scan_positions_a=scan_positions)',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold, voltages_gold, defocuses_gold, '
                   'n_values_gold, group_factors_gold, contrast_floor=0.122, '
                   'scan_positions_a=scan_positions_gold)'},
     {'setup': 'import numpy as np\n'
               'candidates=np.array([[14.0,2.0,7.0,13.0,20.0],[10.0,1.0,6.0,14.0,24.0],[18.0,4.0,10.0,16.0,22.0],[22.0,6.0,12.0,18.0,24.0],[24.0,8.0,14.0,22.0,30.0],[12.0,3.0,9.0,16.0,22.0],[16.0,2.0,8.0,14.0,22.0],[20.0,4.0,10.0,16.0,24.0],[13.0,3.0,8.0,15.0,23.0],[28.0,8.0,14.0,22.0,30.0]],dtype=float)\n'
               'voltages=np.array([100.0,150.0,200.0,300.0],dtype=float)\n'
               'defocuses=np.array([-45.0,-15.0,15.0,45.0],dtype=float)\n'
               'n_values=np.array([16,32],dtype=int)\n'
               'group_factors=np.array([1,2,3,6],dtype=int)\n'
               'scan_positions=np.array([[0.0,0.0],[0.18,0.0],[0.0,0.18],[0.18,0.18],[0.31,-0.13]],dtype=float)\n'
               'import copy\n'
               'candidates_gold = copy.deepcopy(candidates)\n'
               'defocuses_gold = copy.deepcopy(defocuses)\n'
               'scan_positions_gold = copy.deepcopy(scan_positions)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quscope_quantum_resource_design(candidates[[1,8]],voltages[[0,3]],defocuses[[0,3]],np.array([32]),np.array([1,2,3]),scan_positions_a=scan_positions[[0,2,4]])',
      'gold_call': '_oracle_quscope_quantum_resource_design(candidates_gold[[1, 8]], voltages_gold[[0, 3]], '
                   'defocuses_gold[[0, 3]], np.array([32]), np.array([1, 2, 3]), '
                   'scan_positions_a=scan_positions_gold[[0, 2, 4]])'}]
