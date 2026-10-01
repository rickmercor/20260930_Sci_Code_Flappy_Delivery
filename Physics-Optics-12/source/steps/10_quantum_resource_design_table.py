"""
Evaluate every candidate-major (candidate,N,group) row over all declared probe positions. Use dx=6.4/N and an ungrouped N=64 classical scan. Across voltage-major defocus scenarios, classes H0 then H1, and scan positions in input order, record axial-Nyquist feasibility, minimum normalized state fidelity, maximum detector-vector relative error with index S*(2*(D*v+d)+h)+scan where D is the defocus count, minimum same-position H1/H0 contrast defined as the unnormalized Euclidean norm ||d_H1-d_H0||_2, and total scan-workload CX. Return columns [candidate_index,N,group,slices,nq,nyquist_ok,min_fidelity,max_error,min_contrast,total_scan_CX,worst_error_index,feasible,N_position,group_position]; feasibility requires fidelity>=0.999999999999, error<=0.02, contrast>=0.121, and valid Nyquist/nonempty masks everywhere. Both candidate and classical-reference paths end immediately after their final grating; every cost is the STEM body count defined in step 09. The synthetic candidate grid, N=64 reference and feasibility rules are task-defined.

The table couples QuScope circuit execution, its classical-twin validation discipline, detector physics, and hardware resources.

Returns
-------
np.ndarray, (C*len(n_values)*len(group_factors),14) real candidate-major design table with the columns documented below.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantum_resource_design_table(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> 'np.ndarray':
    """Evaluate every candidate, grid, and grouping over the full scan tensor.

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
result : np.ndarray
    (C*len(n_values)*len(group_factors),14) real candidate-major design table with the columns documented below.

Notes
-----
Evaluate every candidate, grid, and grouping over the full scan tensor.

Array-like inputs retain their declared order and scalars set sampling and
admissibility; Cmin is min ||d_H1-d_H0||_2 without normalization. Returns shape
(len(candidates)*len(n_values)*len(group_factors),14), with columns
[candidate,N,g,slices,nq,nyquist,Fmin,Eref,Cmin,CX,error_index,feasible,
N_position,g_position]."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_quantum_resource_design_table(candidates: 'np.ndarray', voltages_kv: 'np.ndarray', defocuses_a: 'np.ndarray', n_values: 'np.ndarray'=(16, 32), group_factors: 'np.ndarray'=(1, 2, 3, 6), field_of_view_a: float=6.4, base_slice_thickness_a: float=1.8, cs_mm: float=0.05, reference_n: int=64, detector_error_tolerance: float=0.02, contrast_floor: float=0.121, fidelity_floor: float=0.999999999999, scan_positions_a: 'np.ndarray'=((0.0, 0.0), (0.18, 0.0), (0.0, 0.18), (0.18, 0.18), (0.31, -0.13))) -> 'np.ndarray':
    """Build the candidate-major QuScope accuracy/resource table."""
    candidates = np.asarray(candidates, dtype=float)
    voltages = np.asarray(voltages_kv, dtype=float)
    defocuses = np.asarray(defocuses_a, dtype=float)
    n_values = np.asarray(n_values, dtype=int)
    group_factors = np.asarray(group_factors, dtype=int)
    scan_positions = np.asarray(scan_positions_a, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 5 or candidates.shape[0] == 0:
        raise ValueError('candidates must have shape (C,5)')
    if voltages.ndim != 1 or voltages.size == 0 or defocuses.ndim != 1 or (defocuses.size == 0):
        raise ValueError('scenario vectors must be nonempty')
    if n_values.ndim != 1 or n_values.size == 0 or group_factors.ndim != 1 or (group_factors.size == 0):
        raise ValueError('design vectors must be nonempty')
    if scan_positions.ndim != 2 or scan_positions.shape[0] < 2 or scan_positions.shape[1] != 2 or (not np.all(np.isfinite(scan_positions))):
        raise ValueError('scan_positions_a must be a finite (S,2) array with S>=2')
    if np.any(np.diff(candidates[:, 1:], axis=1) <= 0.0) or np.any(candidates[:, 0] <= 0.0):
        raise ValueError('candidate angles must be ordered and positive')
    if any((int(n) < 2 or int(n) & int(n) - 1 for n in n_values)):
        raise ValueError('n values must be powers of two')
    if any((int(g) not in (1, 2, 3, 6) for g in group_factors)):
        raise ValueError('group factors must divide six')
    if int(reference_n) < max(n_values) or int(reference_n) & int(reference_n) - 1:
        raise ValueError('reference_n must be a power of two no smaller than all candidates')
    scalars = [field_of_view_a, base_slice_thickness_a, detector_error_tolerance, contrast_floor, fidelity_floor]
    if any((not math.isfinite(float(x)) for x in scalars)):
        raise ValueError('scalar controls must be finite')
    if field_of_view_a <= 0.0 or base_slice_thickness_a <= 0.0 or detector_error_tolerance <= 0.0 or (contrast_floor < 0.0) or (not 0.0 < fidelity_floor <= 1.0):
        raise ValueError('invalid scalar controls')

    def _constants(voltage_kv):
        value = _oracle_relativistic_electron_constants(voltage_kv)
        return (float(value[0]), float(value[1]))

    def _grouped_stack(n, hypothesis, group):
        return _oracle_build_grouped_strong_scattering_stack(n, field_of_view_a / n, hypothesis, group)

    def _probe(n, dx, wavelength, alpha, defocus, position):
        return _oracle_amplitude_encoded_stem_probe(n, dx, wavelength, alpha, defocus, cs_mm, position)

    def _propagator_grid(n, dx, wavelength, distance):
        frequency = np.fft.fftfreq(n, d=dx)
        (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
        return np.exp(-1j * np.pi * wavelength * distance * (kx * kx + ky * ky))

    def _quantum_exit(stack, incident, wavelength, sigma, distance, dx):
        group = int(round(float(distance) / float(base_slice_thickness_a)))
        return _oracle_quantum_multislice_exit_state(stack, incident, wavelength, sigma, base_slice_thickness_a, group, dx)

    def _classical_exit(stack, incident, wavelength, sigma, distance, dx):
        propagation = _propagator_grid(incident.shape[0], dx, wavelength, distance)
        state = incident.copy()
        for (j, potential) in enumerate(stack):
            state *= np.exp(1j * sigma * potential)
            if j < len(stack) - 1:
                state = np.fft.ifft2(np.fft.fft2(state, norm='ortho') * propagation, norm='ortho')
        return state

    def _detector(state, wavelength, dx, edges):
        try:
            return _oracle_annular_detector_vector(state, wavelength, dx, edges)
        except ValueError:
            return None

    def _resources(n, slices):
        value = _oracle_transpiled_multislice_resources(n, slices)
        return tuple((float(x) for x in value))
    reference_dx = field_of_view_a / int(reference_n)
    references = {}
    for (candidate_index, candidate) in enumerate(candidates):
        alpha = float(candidate[0])
        edges = candidate[1:]
        for (voltage_index, voltage) in enumerate(voltages):
            (wavelength, sigma) = _constants(voltage)
            for (defocus_index, defocus) in enumerate(defocuses):
                for (scan_index, position) in enumerate(scan_positions):
                    incident = _probe(int(reference_n), reference_dx, wavelength, alpha, float(defocus), position)
                    for hypothesis in (0, 1):
                        stack = _grouped_stack(int(reference_n), hypothesis, 1)
                        state = _classical_exit(stack, incident, wavelength, sigma, base_slice_thickness_a, reference_dx)
                        references[candidate_index, voltage_index, defocus_index, hypothesis, scan_index] = _detector(state, wavelength, reference_dx, edges)
    rows = []
    for (candidate_index, candidate) in enumerate(candidates):
        alpha = float(candidate[0])
        edges = candidate[1:]
        for (n_position, n) in enumerate(n_values):
            n = int(n)
            dx = field_of_view_a / n
            for (group_position, group) in enumerate(group_factors):
                group = int(group)
                slice_count = 6 // group
                (nq, diagonals, qft2, per_scan_cx) = _resources(n, slice_count)
                cx = per_scan_cx * scan_positions.shape[0]
                nyquist_feasible = True
                minimum_fidelity = math.inf
                maximum_error = -math.inf
                minimum_contrast = math.inf
                worst_index = -1
                for (voltage_index, voltage) in enumerate(voltages):
                    (wavelength, sigma) = _constants(voltage)
                    nyquist = 1000.0 * wavelength / (2.0 * dx)
                    if alpha > nyquist + 1e-12 or edges[-1] > nyquist + 1e-12:
                        nyquist_feasible = False
                    for (defocus_index, defocus) in enumerate(defocuses):
                        for (scan_index, position) in enumerate(scan_positions):
                            incident = _probe(n, dx, wavelength, alpha, float(defocus), position)
                            detector_vectors = []
                            for hypothesis in (0, 1):
                                stack = _grouped_stack(n, hypothesis, group)
                                quantum = _quantum_exit(stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                                classical = _classical_exit(stack, incident, wavelength, sigma, base_slice_thickness_a * group, dx)
                                denominator = float(np.vdot(quantum, quantum).real) * float(np.vdot(classical, classical).real)
                                fidelity = float(abs(np.vdot(quantum, classical)) ** 2 / denominator)
                                minimum_fidelity = min(minimum_fidelity, fidelity)
                                value = _detector(quantum, wavelength, dx, edges)
                                if value is None:
                                    nyquist_feasible = False
                                    value = np.zeros(3, dtype=float)
                                detector_vectors.append(value)
                                reference = references[candidate_index, voltage_index, defocus_index, hypothesis, scan_index]
                                error = float(np.linalg.norm(value - reference) / np.linalg.norm(reference))
                                error_index = scan_positions.shape[0] * (2 * (voltage_index * defocuses.size + defocus_index) + hypothesis) + scan_index
                                if error > maximum_error:
                                    maximum_error = error
                                    worst_index = error_index
                            contrast = float(np.linalg.norm(detector_vectors[1] - detector_vectors[0]))
                            minimum_contrast = min(minimum_contrast, contrast)
                feasible = nyquist_feasible and minimum_fidelity >= fidelity_floor and (maximum_error <= detector_error_tolerance) and (minimum_contrast >= contrast_floor)
                rows.append([float(candidate_index), float(n), float(group), float(slice_count), float(nq), float(nyquist_feasible), float(minimum_fidelity), float(maximum_error), float(minimum_contrast), float(cx), float(worst_index), float(feasible), float(n_position), float(group_position)])
    return np.asarray(rows, dtype=float)

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
      'call': 'quantum_resource_design_table(candidates,voltages,defocuses,n_values,group_factors)',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold, voltages_gold, defocuses_gold, '
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
               'n_values_gold = copy.deepcopy(n_values)\n'
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quantum_resource_design_table(candidates[[1,5]],voltages[[0,2]],defocuses[[0,2]],n_values,np.array([2,3]))',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[1, 5]], voltages_gold[[0, 2]], '
                   'defocuses_gold[[0, 2]], n_values_gold, np.array([2, 3]))'},
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
               'voltages_gold = copy.deepcopy(voltages)',
      'call': 'quantum_resource_design_table(candidates[[0,4]],voltages,defocuses,np.array([16,32]),np.array([1,6]))',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[0, 4]], voltages_gold, '
                   'defocuses_gold, np.array([16, 32]), np.array([1, 6]))'},
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
      'call': 'quantum_resource_design_table(candidates[[1]],voltages[[0,2]],defocuses[[0,2]],np.array([32]),np.array([2,3]),scan_positions_a=scan_positions[[0,3]])',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[1]], voltages_gold[[0, 2]], '
                   'defocuses_gold[[0, 2]], np.array([32]), np.array([2, 3]), '
                   'scan_positions_a=scan_positions_gold[[0, 3]])'},
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
      'call': 'quantum_resource_design_table(candidates[[1]],voltages[[0,3]],defocuses[[0,3]],np.array([32]),np.array([1,2]),scan_positions_a=scan_positions[[0,4]])',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[1]], voltages_gold[[0, 3]], '
                   'defocuses_gold[[0, 3]], np.array([32]), np.array([1, 2]), '
                   'scan_positions_a=scan_positions_gold[[0, 4]])'},
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
      'call': 'quantum_resource_design_table(candidates[[5,8]],voltages[[1,2]],defocuses[[1,2]],np.array([16,32]),np.array([1]),scan_positions_a=scan_positions[[1,2,3]])',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[5, 8]], voltages_gold[[1, 2]], '
                   'defocuses_gold[[1, 2]], np.array([16, 32]), np.array([1]), '
                   'scan_positions_a=scan_positions_gold[[1, 2, 3]])'},
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
      'call': 'quantum_resource_design_table(candidates[[9]],voltages[[2,3]],defocuses[[0,3]],np.array([16,32]),np.array([1,6]),scan_positions_a=scan_positions[[0,4]])',
      'gold_call': '_oracle_quantum_resource_design_table(candidates_gold[[9]], voltages_gold[[2, 3]], '
                   'defocuses_gold[[0, 3]], np.array([16, 32]), np.array([1, 6]), '
                   'scan_positions_a=scan_positions_gold[[0, 4]])'}]
