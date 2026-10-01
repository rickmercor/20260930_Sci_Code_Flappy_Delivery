# Physics-Optics-12

## Background

Coherent electron scattering links the specimen potential, probe optics and measured detector signal. Numerical agreement between two implementations and convergence with spatial or depth resolution answer different scientific questions. Circuit resource estimates provide a complementary measure of the cost of representing this evolution.

## Problem

Determine the admissible minimum-CX amplitude-encoded multislice STEM design for the following task-defined benchmark, and report its dimensionless cost–distortion audit score R. The fixed rows [alpha,e0,e1,e2,e3] in mrad are [[14,2,7,13,20],[10,1,6,14,24],[18,4,10,16,22],[22,6,12,18,24],[24,8,14,22,30],[12,3,9,16,22],[16,2,8,14,22],[20,4,10,16,24],[13,3,8,15,23],[28,8,14,22,30]]; use voltages [100,150,200,300] kV, defocuses [-45,-15,15,45] Å, grids [16,32], grouping factors [1,2,3,6], and scan positions [(0,0),(0.18,0),(0,0.18),(0.18,0.18),(0.31,-0.13)] Å in the listed orders on a periodic L=6.4 Å cell with array axis 0=x and axis 1=y. Each class has six base slices of thickness 1.8 Å, indexed j=0,...,5, with projected potentials in V Å formed from A*exp(-r_periodic²/(2w²)) at centers (0,0), (L/2,L/2), (L/4,3L/4), using shortest wrapped coordinate displacements in [-L/2,L/2), widths [0.42,1.25*0.42,0.9*0.42] Å, central amplitudes (1500 for H1, 1050 for H0)*(1+0.08*((j mod 3)-1)), and other amplitudes 650*(1+0.05*(-1)^j), 400*(1-0.04*(-1)^j); a retained grating sums each consecutive group of g projected potentials, with a propagation distance of 1.8g Å between retained gratings and the exit immediately after the final grating. Use m0=9.1093837139e-31 kg, h=6.62607015e-34 J s, e=1.602176634e-19 C, c=299792458 m/s, Cs=0.05 mm, orthonormal transforms with forward kernel exp(-2*pi*i*k·r), unshifted frequencies, three detector channels with half-open angular intervals [e0,e1), [e1,e2), [e2,e3), and exact exponential transmission.

For every candidate/grid/grouping, admissibility requires axial-Nyquist support and nonempty detector masks at every voltage, minimum normalized quantum/classical state fidelity at matching grids and grouping >=0.999999999999, maximum detector-vector relative error E_ref against the ungrouped N=64 reference <=0.02, and minimum same-position contrast C_min=min ||d_H1-d_H0||_2 >=0.121, where the maxima/minima cover all listed voltages, defocuses, hypotheses and scan positions, and E_ref uses the reference-vector norm as denominator. Minimize (five-position CX, E_ref, -C_min, zero-based candidate-row index, grid order, grouping order, stable order), costing only the multislice probe-circuit body with all-to-all connectivity, arbitrary-diagonal synthesis and full-register QFT swaps; state preparation and the final detector-plane transform are outside this resource boundary. Define the absolute angular margin M=min_voltage[1000*lambda/(2*dx)-max(alpha,e3)] mrad with dx=L/N and lambda in Å, fix theta0=1 mrad and mu=M/theta0, and set f_D=diagonal_CX/total_CX; the task-defined linearization diagnostic is E_lin=max ||d_lin-d_exact||_2/||d_exact||_2 using 1+i*sigma*V at each retained grating, the same between-grating propagation, and one normalization of the complete linearized exit state. For the selected design, the task-defined dimensionless audit is R=CX*E_ref*E_lin/(C_min*mu*f_D), rounded only once as round(R,6).

In <reasoning>, justify the electron-optical construction and the resource-minimizing decision mathematically, interpret the factors of R, and report selected_design=(candidate,N,g), CX, E_ref, C_min, E_lin, M with units, f_D, and Phi=max|sigma*V| over the selected retained gratings and voltages with its task-defined ratio to the paper's 1.7-rad example. Include the design and affected diagnostic for each single-constraint relaxation (contrast floor or reference-error ceiling), the grouping comparison G=max ||d_selected,g=3-d_selected,g=1||_2/||d_selected,g=1||_2 over the listed operating conditions on the selected candidate/grid, and the detector-observable and CX-count effects of an extra final free-propagation segment. Identify the physical benchmark underlying the paper's strong-scattering phase example, and explain the paper's finite-grid HAADF collection example, including its effective angular range and effect on the reported signal. Report diagnostic decimals to at least eight significant digits (absolute tolerance 5e-7 for the tagged scalar, 5e-8 for diagnostic decimals; resource counts and design indices are exact).

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Include the derivations and source-based explanations requested above. Show only the scalars this prompt asks you to report.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Include the derivations and source-based explanations requested above. Show only the scalars this prompt asks you to report.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

relativistic_electron_constants

Goal
----
Compute the paper's relativistic wavelength and interaction constant. Use V=1000*voltage_kv volts, lambda=h/sqrt(2*m0*e*V*(1+e*V/(2*m0*c^2)))*1e10 Å, and sigma=2*pi/(lambda*V)*(m0*c^2+e*V)/(2*m0*c^2+e*V), with h=6.62607015e-34, m0=9.1093837139e-31, e=1.602176634e-19, and c=299792458. Reject nonfinite or nonpositive voltage.

```python
def relativistic_electron_constants(voltage_kv: float) -> 'np.ndarray':
    """Convert one positive finite accelerating voltage in kV.

Parameters
----------
voltage_kv : float
    Positive finite accelerating voltage, in kV.

Returns
-------
result : np.ndarray
    (2,) float64 vector [wavelength_A, interaction_constant_rad_per_VA].

Notes
-----
Convert one positive finite accelerating voltage in kV.

Returns a float64 ndarray of shape (2,) ordered as
[wavelength_A, interaction_constant_rad_per_VA].

Raises ValueError for nonfinite or nonpositive voltage_kv."""
    return result
```

### Step 2

build_grouped_strong_scattering_stack

Goal
----
Construct the explicit six-slice periodic benchmark, then sum consecutive groups. On L=N*dx, wrap each coordinate displacement to [-L/2,L/2). Axis 0 is x and axis 1 is y: a center (cx,cy) sits at array index [cx/pixel_size_a,cy/pixel_size_a] modulo N, equivalently meshgrid indexing='ij'. Use centers (0,0), (L/2,L/2), (L/4,3L/4), w=0.42 Å, and A*exp(-r_periodic^2/(2*width^2)). In slice j=0..5, the central amplitude is (1500 for hypothesis 1, else 1050)*(1+0.08*((j%3)-1)); the other amplitudes are 650*(1+0.05*(-1)^j) and 400*(1-0.04*(-1)^j), with widths w,1.25w,0.9w. For g in {1,2,3,6}, return 6/g slices formed by summing base[j:j+g].

```python
def build_grouped_strong_scattering_stack(n: int, pixel_size_a: float, hypothesis: int, group_factor: int=1) -> 'np.ndarray':
    """Build the six-slice periodic potential for one specimen hypothesis.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
hypothesis : int
    Specimen class: 0 denotes H0 and 1 denotes H1.
group_factor : int
    Number of consecutive base slices per retained grating; one of 1, 2, 3, 6.

Returns
-------
result : np.ndarray
    (6//group_factor,N,N) real grouped projected potentials in V Å.

Notes
-----
Build the six-slice periodic potential for one specimen hypothesis.

n and pixel_size_a define the square grid; hypothesis is class H0 (0) or
class H1 (1); group_factor must divide six. Axis 0 is x and axis 1 is y,
so (cx,cy) maps to [cx/pixel_size_a,cy/pixel_size_a] modulo n. Returns a
float ndarray of shape (6//group_factor,n,n), with consecutive slices
summed in order."""
    return result
```

### Step 3

amplitude_encoded_stem_probe

Goal
----
Form a displaced QuScope STEM input state on two log2(N)-qubit registers. With unshifted fftfreq and ij indexing, use A(k)=1 for |k|<=alpha*1e-3/lambda+1e-15; chi=pi*lambda*defocus*k^2+(pi/2)*(cs_mm*1e7)*lambda^3*k^4; multiply by exp(-2*pi*i*(kx*x_scan+ky*y_scan)), compute the orthonormal IFFT2, then normalize once. Array entry [a,b] is the amplitude of |a>row tensor |b>column in row-major flattening.

```python
def amplitude_encoded_stem_probe(n: int, pixel_size_a: float, wavelength_a: float, convergence_mrad: float, defocus_a: float, cs_mm: float=0.05, scan_position_a: tuple=(0.0, 0.0)) -> 'np.ndarray':
    """Prepare one displaced aberrated STEM probe on an n-by-n grid.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
wavelength_a : float
    Positive finite electron wavelength, in Å.
convergence_mrad : float
    Positive finite circular-aperture semi-angle, in mrad.
defocus_a : float
    Finite defocus, in Å; either sign is allowed.
cs_mm : float
    Finite nonnegative spherical-aberration coefficient, in mm.
scan_position_a : tuple
    Finite (x,y) probe displacement, in Å.

Returns
-------
result : np.ndarray
    (N,N) unit-L2-normalized complex incident amplitudes.

Notes
-----
Prepare one displaced aberrated STEM probe on an n-by-n grid.

Sampling and optical arguments are finite scalars and scan_position_a is
an (x,y) pair in angstrom. Returns a normalized complex ndarray of shape
(n,n), using the stated Fourier sign and register order."""
    return result
```

### Step 4

quantum_slice_block

Goal
----
Apply the exact position-space phase grating D[exp(i*sigma*V)]. If propagate_after is True, also apply the separable orthonormal forward transform F@state@F.T with F[a,b]=exp(-2*pi*i*a*b/N)/sqrt(N), the reciprocal phase exp(-i*pi*lambda*Delta_z*(kx^2+ky^2)) on the unshifted ij-indexed grid, and the inverse transform. If False, omit the whole propagation segment. The public optional flag makes interior and terminal STEM gratings explicit; projected potentials are in V Å.

```python
def quantum_slice_block(state: 'np.ndarray', projected_potential: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float, propagate_after: bool=True) -> 'np.ndarray':
    """Apply an exact phase grating and an optional following Fresnel segment.

Parameters
----------
state : array_like
    Finite complex (N,N) state on a power-of-two grid.
projected_potential : array_like
    Finite real (N,N) projected potential, in V Å.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Positive finite interaction constant in rad/(V Å).
propagation_distance_a : float
    Positive finite inter-grating propagation distance, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
propagate_after : bool
    Whether to include the free-propagation segment after this transmission.

Returns
-------
result : np.ndarray
    (N,N) complex amplitudes after transmission and the optional Fresnel segment.

Notes
-----
Apply an exact phase grating and an optional following Fresnel segment.

state and projected_potential are equal (N,N) power-of-two arrays; potential
is projected in V Å. wavelength_a, propagation_distance_a, pixel_size_a
are positive finite Å scalars, and interaction_constant is positive finite
rad/(V Å). Boolean propagate_after=True includes the following segment;
False returns immediately after the grating, as for the final STEM slice.
Returns a complex ndarray of the same (N,N) shape as state. Invalid inputs raise ValueError."""
    return result
```

### Step 5

quantum_multislice_exit_state

Goal
----
Execute quantum_slice_block once for each consecutive grouped potential, in order. Use propagation distance base_slice_thickness_a*group_factor and propagate_after=True only when another retained grating follows; use False for the final grating. The stack contains exactly 6/group_factor slices. Return the immediate post-grating STEM exit, without an objective lens or final free-propagation segment.

```python
def quantum_multislice_exit_state(grouped_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, base_slice_thickness_a: float, group_factor: int, pixel_size_a: float) -> 'np.ndarray':
    """Propagate an incident state through every grouped slice in order.

Parameters
----------
grouped_stack : array_like
    Finite real (s,N,N) projected potentials in V Å, with s*group_factor=6.
incident_state : array_like
    Finite complex (N,N) input amplitudes matching the potential grid.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Positive finite interaction constant in rad/(V Å).
base_slice_thickness_a : float
    Positive finite thickness of one base slice, in Å.
group_factor : int
    Number of consecutive base slices per retained grating; one of 1, 2, 3, 6.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.

Returns
-------
result : np.ndarray
    (N,N) complex amplitudes immediately after the final retained grating.

Notes
-----
Propagate an incident state through every grouped slice in order.

grouped_stack is (s,n,n), incident_state is (n,n), and s*group_factor
equals six. Propagation is applied only between gratings, not after the
final one; scalar arguments are positive and finite. Returns the final
complex ndarray with the same (n,n) shape as incident_state."""
    return result
```

### Step 6

annular_detector_vector

Goal
----
Transform the exit amplitudes to momentum space with orthonormal FFT2, form normalized intensity |psi(k)|^2/||psi||^2, and sum it over the three consecutive half-open annuli [e0,e1), [e1,e2), [e2,e3). Use theta_mrad=1000*lambda*sqrt(kx^2+ky^2), unshifted fftfreq, ij indexing, and reject an empty mask.

```python
def annular_detector_vector(exit_state: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Integrate reciprocal intensity in three consecutive half-open annuli.

Parameters
----------
exit_state : array_like
    Finite nonzero square complex exit amplitudes; no unit-norm precondition.
wavelength_a : float
    Positive finite electron wavelength, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
detector_edges_mrad : array_like
    Four finite increasing nonnegative angles in mrad; each half-open annulus must contain a grid point.

Returns
-------
result : np.ndarray
    (3,) real probabilities for the three half-open detector annuli.

Notes
-----
Integrate reciprocal intensity in three consecutive half-open annuli.

exit_state is a finite, nonzero square complex array; wavelength_a and pixel_size_a are positive; and
detector_edges_mrad contains four increasing edges with nonempty masks.
Probabilities use the full state norm, so the input need not be unit norm.
Invalid sampling, edges or empty masks raise ValueError.
Returns a float ndarray [P0,P1,P2] of shape (3,)."""
    return result
```

### Step 7

bivariate_multislice_orders

Goal
----
Resolve the task-defined linearized transmission model by scattering origin.

For two projected-potential stacks B and D, define a formal exit wave by using
the grating 1+i*sigma*(a*B_j+b*D_j) at each slice, and the declared Fresnel
propagation between successive gratings. Return the complex coefficients of
this bivariate polynomial in a and b at the immediate post-final-grating plane.
These are polynomial coefficients, not derivatives or probabilities. This
diagnostic is derived from QuScope's ordered coherent propagation; it is not a
separate simulation feature claimed to ship in QuScope.

```python
def bivariate_multislice_orders(background_stack: 'np.ndarray', contrast_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float) -> 'np.ndarray':
    """Return the unnormalized bivariate scattering-order exit amplitudes.

Parameters
----------
background_stack : array_like
    Finite real (s,N,N) projected background potentials B, in V Å; 1<=s<=6.
contrast_stack : array_like
    Finite real projected contrast potentials D, in V Å, with the same shape as B.
incident_state : array_like
    Finite complex (N,N) input amplitudes matching the potential grid. It must be nonzero and is used without renormalization.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Finite nonnegative interaction constant in rad/(V Å).
propagation_distance_a : float
    Positive finite inter-grating propagation distance, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.

Returns
-------
result : np.ndarray
    ((s+1)*(s+2)//2,N,N) complex unnormalized bivariate coefficients in the documented triangular order.

Notes
-----
Return the unnormalized bivariate scattering-order exit amplitudes.

background_stack B and contrast_stack D: finite real arrays (s,N,N),
1<=s<=6, with projected potentials in V Å. N>=2 is a power of two;
within each (N,N) slice, axis 0=x and axis 1=y. incident_state: finite nonzero complex
(N,N) amplitude array, used without renormalization. wavelength_a,
propagation_distance_a and pixel_size_a: positive finite Å scalars.
interaction_constant: finite nonnegative rad/(V Å).

For formal dimensionless a,b, each grating is
1+i*interaction_constant*(a*B_j+b*D_j). The between-grating Fresnel
operator has reciprocal phase exp(-i*pi*wavelength_a*distance*k^2),
unshifted frequencies and orthonormal forward exp(-2*pi*i*k.r).
The exit follows the final grating immediately, with no terminal gap.

Returns complex128 (K,N,N), K=(s+1)*(s+2)//2. The coefficient of
a**p*b**q has row k=d*(d+1)//2+q, d=p+q<=s. No coefficient or state is
normalized; the output polynomial equals the complete ordered exit.
Invalid shapes, ranges or nonfinite data raise ValueError."""
    return result
```

### Step 8

annular_order_coherence

Goal
----
Return the detector-restricted coherence of scattering-order amplitudes.

The order labels describe coherent alternatives, not an incoherent ensemble.
The requested matrices retain their complex cross terms and their absolute
norms. They permit the normalized detector probability of any reconstructed
exit wave to be evaluated against the full-space norm, including interference.
This is a task-derived representation of the paper's momentum-plane observable.

```python
def annular_order_coherence(order_amplitudes: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Return unnormalized detector-restricted complex coherence matrices.

Parameters
----------
order_amplitudes : array_like
    Finite complex (K,N,N) coefficient amplitudes; K>=1, N>=2 a power of two. Zero coefficients are allowed.
wavelength_a : float
    Positive finite electron wavelength, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
detector_edges_mrad : array_like
    Four finite increasing nonnegative angles in mrad; each half-open annulus must contain a grid point.

Returns
-------
result : np.ndarray
    (4,K,K) complex unnormalized coherence matrices for three annuli and the full reciprocal grid.

Notes
-----
Return unnormalized detector-restricted complex coherence matrices.

order_amplitudes: finite complex (K,N,N), K>=1, N>=2 a power of two.
These are amplitude coefficients, with neither independent nor joint
normalization imposed. Zero coefficient arrays are valid.
wavelength_a and pixel_size_a: positive finite Å scalars.
detector_edges_mrad: four finite strictly increasing nonnegative angles,
defining three consecutive half-open annuli. Every mask must be nonempty.
Within each (N,N) coefficient, axis 0=x and axis 1=y; these are the last
two input axes. Use unshifted frequencies and an orthonormal forward
transform with kernel exp(-2*pi*i*k.r).

Returns complex128 (4,K,K). If z_a is the transformed coefficient a,
H[c,a,b]=sum_mask_c(conj(z_a)*z_b). Channels 0,1,2 are the annuli;
channel 3 is the entire reciprocal grid, not just their union. There is
no normalization or deletion of complex off-diagonal entries. For a
coherent superposition with coefficient weights w, its probability is
(w.conj() @ H[c] @ w)/(w.conj() @ H[3] @ w).
Invalid shapes, sampling, nonfinite data or empty masks raise ValueError."""
    return result
```

### Step 9

transpiled_multislice_resources

Goal
----
Cost the released STEM probe-circuit body for an N-by-N grid and s retained gratings. With q=log2(N), nq=2q, count s transmission diagonals and s-1 propagation diagonals, each at 2^nq-2 CX, and 2(s-1) separable two-dimensional Fourier transforms. A full q-qubit register QFT costs q(q-1)+3*floor(q/2) CX, including swaps, and each two-dimensional transform acts on both registers. This all-to-all analytic synthesis convention reproduces the paper's STEM resource examples; state preparation, detector-plane readout and device routing are outside the boundary. Return [nq,2s-1,2(s-1),total_CX].

```python
def transpiled_multislice_resources(n: int, slice_count: int) -> 'np.ndarray':
    """Count the all-to-all STEM probe-circuit body, with no final propagation.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
slice_count : int
    Positive integer retained-grating count.

Returns
-------
result : np.ndarray
    (4,) real vector [logical_qubits,diagonal_gate_count,separable_2D_QFT_count,total_CX].

Notes
-----
Count the all-to-all STEM probe-circuit body, with no final propagation.

n is the linear grid dimension and slice_count is the positive retained
grating count. State preparation and detector-plane readout are outside
this cost boundary. Returns a float ndarray of shape (4,) ordered as
[logical_qubits,diagonal_gate_count,separable_2D_QFT_count,total_CX]."""
    return result
```

### Step 10

quantum_resource_design_table

Goal
----
Evaluate every candidate-major (candidate,N,group) row over all declared probe positions. Use dx=6.4/N and an ungrouped N=64 classical scan. Across voltage-major defocus scenarios, classes H0 then H1, and scan positions in input order, record axial-Nyquist feasibility, minimum normalized state fidelity, maximum detector-vector relative error with index S*(2*(D*v+d)+h)+scan where D is the defocus count, minimum same-position H1/H0 contrast defined as the unnormalized Euclidean norm ||d_H1-d_H0||_2, and total scan-workload CX. Return columns [candidate_index,N,group,slices,nq,nyquist_ok,min_fidelity,max_error,min_contrast,total_scan_CX,worst_error_index,feasible,N_position,group_position]; feasibility requires fidelity>=0.999999999999, error<=0.02, contrast>=0.121, and valid Nyquist/nonempty masks everywhere. Both candidate and classical-reference paths end immediately after their final grating; every cost is the STEM body count defined in step 09. The synthetic candidate grid, N=64 reference and feasibility rules are task-defined.

```python
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
```

### Step 11

select_quantum_resource_design

Goal
----
Filter column 11 for feasible rows, then select the lexicographic minimum by (CX column 9, max error column 7, negative contrast column 8, candidate index, N_position, group_position, stable row index). Return [row_index,candidate_index,N,group,slices,nq,worst_error_index,min_fidelity,max_error,min_contrast,CX]. Raise ValueError if no row is feasible.

```python
def select_quantum_resource_design(design_table: 'np.ndarray') -> 'np.ndarray':
    """Select one feasible row using the exact lexicographic resource key.

Parameters
----------
design_table : array_like
    Finite nonempty float array-like (R,14) with the documented design-table columns.

Returns
-------
result : np.ndarray
    (11,) real selected-design certificate [row_index,candidate,N,g,slices,nq,error_index,Fmin,Eref,Cmin,CX].

Notes
-----
Select one feasible row using the exact lexicographic resource key.

design_table has shape (R,14). Returns a float ndarray of shape (11,)
ordered as [row_index,candidate,N,g,slices,nq,error_index,Fmin,Eref,
Cmin,CX]; stable input order resolves an exact tie.
Raises ValueError if the table is empty, nonfinite, has a shape other than
(R,14), or contains no feasible row."""
    return result
```

### Step 12

quscope_quantum_resource_design

Goal
----
Compose every preceding public function to select the task-defined admissible STEM design. Exact and linearized paths apply propagation only between retained gratings. For the selected row, compare the once-normalized full linearized exit with the exact exit over the complete voltage/defocus/hypothesis/position scan, using E_lin=max ||d_lin-d_exact||_2/||d_exact||_2. Define M=min_voltage[1000*lambda/(2*dx)-max(alpha,outer_edge)] mrad, theta0=1 mrad and mu=M/theta0. With f_D=scan_count*(2*slices-1)*(2^logical_qubits-2)/total_scan_CX, return round(total_scan_CX*E_ref*E_lin/(C_min*mu*f_D),6). R is a dimensionless task-authored post-selection audit, not a published QuScope quantity. Resolve the linearized wave with bivariate_multislice_orders using B=H0 and D=H1-H0, and obtain its detector probability from annular_order_coherence at (a,b)=(1,0) or (1,1), respectively. The full-space coherence channel supplies the normalization of the complete reconstructed exit. This coherent order representation is an exact re-expression of the existing linearized model, not an additional approximation or a new resource charge.

```python
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
```
