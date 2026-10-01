# Certify a constrained optical sampling design

## Background

A boundary-driven optical resonator retains phase-bearing state across propagation delays, so transient observables need a nonstationary treatment. The published sources define the relevant optical and uncertainty constructions; their equations and procedural ordering are not repeated here.

The numerical configuration and author-defined interface conventions fix a deterministic synthetic certification problem. Invalid shapes, nonfinite inputs, nonpositive physical scales, an unavailable crossing, an insufficient calibration grid, use of an approximation outside its stated validity regime, or an empty feasible set raise `ValueError`.

## Problem

Using the relevant published literature, certify a boundary-driven two-mirror resonator with `c=299792458.0 m/s`, `L=2.75 m`, `lambda=1064e-9 m`, intensity reflectivities `(Ra,Rb)=(0.985,0.9995)`, finesse `F=400.0`, phase setting `gamma=0.25 rad`, `n=20001` centered physical round trips, ordered signed operating multipliers `[-1.25,-0.8,0.65,1.1]`, and ordered requested rates `[120000,180000,260000,360000,430000,520000,600000,700000,1000000,1100000] Hz`.

The remaining instance data are a cavity-length drive `delta_d_j = m*v_cr*t_j + (0.18*lambda/F)*sin(2*pi*18000 Hz*t_j + 0.35 rad)` at each physical round-trip time `t_j`, with no additional mirror retardation, where `m` is the operating multiplier, `v_cr` is the critical velocity, and `v_cr` and `F` are evaluated at each covariance node; cosine input-amplitude modulation `(0.12,11000 Hz)`; sine input-phase modulation `(0.18 rad,23000 Hz)`; zero pre-trace state; ordered tolerances `[0.20,1.5e-6 s,0.55,0.10,0.04]`; a maximum retained stride of `350` physical round trips; a separate unit-input, purely linear, unmodulated centered calibration trace on the inclusive interval from `4` through `24` storage times, the storage time being the node cavity's 1/e field-amplitude decay time computed from the node reflectivities; covariance `[[0.01,0.007,-0.00125],[0.007,0.04,0.002],[-0.00125,0.002,0.0025]]` in `[logit(Ra),logit(Rb),ln(F)]`; and one-sided factor `1.645`.

For stride `N`, retain indices `0,N,2N,...` without appending the endpoint and truncate the full reference at the last retained time. On those matched intervals define four diagnostics: relative peak-power error using the truncated full-grid peak as denominator; absolute error between the linearly interpolated full-grid PDH zero nearest the full-grid power peak and the retained-grid PDH zero nearest that reference zero; PDH NRMSE after linearly interpolating the retained trace onto the truncated full grid, normalized by the RMS of the full-grid PDH trace; and the Euclidean error of the two complex PDH coefficients `C(f) = sum_k w_k*(V_k - mu_w)*exp(-2*pi*i*f*t_k) / sum_k w_k` at `18000` and `23000 Hz`, normalized by the Euclidean norm of the full-grid coefficient pair, where on each grid `w_k = sin^2(pi*(t_k - t_first)/(t_last - t_first))` is the symmetric Hann window over that grid's own times and `mu_w` is the window-weighted mean of that grid's PDH trace.

Define the fifth diagnostic by fitting the continuum-tail incident amplitude independently on the full and retained calibration samples using complex least squares, converting each fitted amplitude to incident power as `|alpha|^2`, and taking `|P_retained-P_full|/P_full`. The full calibration samples are all calibration-trace samples inside the window; the retained samples are those at the stride indices `0,N,2N,...` of the whole calibration trace that fall inside it.

Propagate the covariance with a seven-node author-defined rule: centre first with weight `1/2`, then, in lower-Cholesky column order, each positive displacement followed by its negative, each with weight `1/12` and scale `sqrt(6)`; map every node back with inverse-logit reflectivities and an exponential finesse. At each physical node take the componentwise maximum of the four diagnostics over all supplied velocities; for the continuum-tail bias use only multipliers satisfying the author convention `|multiplier| >= 1`, retain the signed velocity in the paper-derived continuum expression, and use NumPy's principal complex square root for the negative branch.

For each candidate and diagnostic, take the nodewise velocity maxima first, then form the weighted population mean and standard deviation across the seven nodes, form `U=mean+1.645*standard_deviation` in physical units, and divide `U` once by its matching tolerance. A candidate is feasible only when every node has a complete non-partial sampling plan, its maximum stride across nodes is at most `350`, and each of the five normalized upper bounds is independently at most one.

Among feasible candidates minimize the unrounded centre-node effective rate, breaking exact ties by original candidate order. Report the selected fifth-diagnostic upper bound in ppm after one multiplication by `10^6` and one rounding to six decimals.

The short reasoning certificate must report the selected requested rate, centre effective rate within `0.5 Hz`, inverse-curve ratio, switch boundary, branch, and stride; the immediately lower-effective requested rate and its fifth normalized bound within `5e-6`; the critical velocity within `5e-10 m/s` and admitted signed tail multipliers; the seven-node covariance reconstruction residual; and the selected fifth-diagnostic population mean, population standard deviation, and unnormalized upper bound within `5e-10`. It must also state the rules and expressions you apply for the sampling regimes and switch boundary, the round-trip field, the PDH signal, the continuum tail, and the fifth-diagnostic, aggregation and feasibility steps.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the rules and scalars the problem statement asks for.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

adaptive_sampling_plan

Goal
----
Recover and implement the paper's adaptive effective-sampling selection across its three physical regimes.

```python
import numpy as np

def adaptive_sampling_plan(desired_hz, length_m, reflectivity_a, reflectivity_b, c_m_s=299792458.0):
    """Return [f_calc, theta, N, n_subhistories, partial_flag, accuracy]: effective sampling rate f_calc in Hz, time step theta = 1/f_calc in s, round trips N per integration step, number of interleaved sub-histories (1 when none), partial_flag 1.0 in the capped partial-update branch and 0.0 otherwise, and accuracy = 1 - |f_calc - desired_hz|/desired_hz."""
    return np.zeros(6, dtype=float)
```

### Step 2

cavity_critical_velocity

Goal
----
Compute the resonance-crossing critical mirror speed from the storage-time criterion in Eq. (4).

```python
def cavity_critical_velocity(wavelength_m, length_m, finesse, c_m_s=299792458.0):
    """Return the positive critical velocity in m/s."""
    return 0.0
```

### Step 3

moving_cavity_drive

Goal
----
Construct the centered round-trip time grid, relative optical-path drive, and amplitude/phase-modulated complex input field from the critical speed supplied by the preceding step.

```python
import numpy as np

def moving_cavity_drive(n_roundtrips, length_m, wavelength_m, critical_velocity_m_s,
                        velocity_multiplier, displacement_amplitude_m,
                        displacement_hz, amplitude_depth, amplitude_hz,
                        phase_depth, phase_hz, phase_offset=0.35,
                        c_m_s=299792458.0):
    """Return the deterministic n-by-4 drive array."""
    return np.empty((0, 4), dtype=float)
```

### Step 4

propagate_cavity_field

Goal
----
Propagate the intracavity field one physical round trip at a time with the paper's delayed recursive phase and a zero stored field before the first sample.

```python
import numpy as np

def propagate_cavity_field(drive, wavelength_m, reflectivity_a, reflectivity_b):
    """Return the complex intracavity field on the round-trip grid."""
    return np.empty(0, dtype=complex)
```

### Step 5

cavity_observables

Goal
----
Convert the complex field into intracavity power and the approximate demodulated PDH signal.

```python
import numpy as np

def cavity_observables(field, drive, demodulation_phase):
    """Return an n-by-2 real array with columns [power, PDH]."""
    return np.empty((0, 2), dtype=float)
```

### Step 6

sampling_diagnostics

Goal
----
Measure the three defined time-domain sampling errors and the phase-sensitive two-tone PDH spectral distortion after retaining every Nth round-trip sample.

```python
import numpy as np

def sampling_diagnostics(time_s, observables, stride, spectral_hz):
    """Return a length-4 float array
    [peak_error, zero_error_s, PDH_NRMSE, spectral_distortion].

    The two columns of observables are power and signed PDH respectively. The retained
    grid uses indices 0,stride,2*stride,...; spectral_hz contains two positive frequencies.
    Invalid shapes, nonfinite data, non-increasing time, an invalid stride, an unavailable
    zero crossing, or a zero normalization denominator raise ValueError.
    """
    return np.zeros(4, dtype=float)
```

### Step 7

ringdown_tail_inference

Goal
----
Infer incident power from the paper's nonadiabatic continuum tail on full and retained grids, then quantify sampling bias.

```python
import numpy as np

def ringdown_tail_inference(time_s, field, stride, velocity_m_s,
                            length_m, wavelength_m, reflectivity_a,
                            reflectivity_b, finesse,
                            tail_start_storage=4.0,
                            tail_end_storage=24.0,
                            c_m_s=299792458.0):
    """Return [P_full_fit,P_retained_fit,relative_bias,n_retained_tail]."""
    return np.zeros(4, dtype=float)
```

### Step 8

transformed_sigma_points

Goal
----
Derive a symmetric seven-point positive rule in correlated logit-reflectivity and log-finesse coordinates, then map every point back to physical optics parameters.

```python
import numpy as np

def transformed_sigma_points(reflectivity_a, reflectivity_b, finesse,
                             transformed_covariance):
    """Return seven rows [Ra_h,Rb_h,F_h,weight_h] in the fixed order."""
    return np.empty((0, 4), dtype=float)
```

### Step 9

uncertainty_risk_table

Goal
----
Risk-load five diagnostic distributions independently and apply completeness, memory, and tolerance filters.

```python
def uncertainty_risk_table(candidate_desired_hz, sigma_sampling_plans,
                           sigma_metrics, sigma_weights, risk_quantile,
                           peak_tolerance, zero_tolerance_s, rms_tolerance,
                           spectral_tolerance, tail_power_tolerance,
                           memory_limit_roundtrips):
    """Return an (number_of_candidates,13) float table with columns
    [desired_hz, centre_f_calc_hz, max_stride,
    peak_bound_normalized, zero_time_bound_normalized,
    nrmse_bound_normalized, spectral_bound_normalized,
    tail_bound_normalized, mean_tail, std_tail, tail_upper_bound,
    composite_bound, feasible] in that order. composite_bound is the largest of
    the five normalized bounds, and feasible is 1.0 or 0.0.
    """
    return np.empty((np.asarray(candidate_desired_hz).size, 13), dtype=float)
```

### Step 10

adaptive_cavity_sampling_benchmark

Goal
----
Compose the discrete moving-cavity diagnostics with the nonadiabatic tail inverse problem and uncertainty certification.

```python
def adaptive_cavity_sampling_benchmark(candidate_desired_hz, velocity_multipliers,
                                        n_roundtrips, length_m, wavelength_m,
                                        reflectivity_a, reflectivity_b, finesse,
                                        demodulation_phase, displacement_fraction,
                                        displacement_hz, amplitude_depth, amplitude_hz,
                                        phase_depth, phase_hz, phase_offset,
                                        peak_tolerance, zero_tolerance_s, rms_tolerance,
                                        spectral_hz, spectral_tolerance,
                                        tail_power_tolerance, memory_limit_roundtrips,
                                        transformed_covariance, risk_quantile,
                                        c_m_s=299792458.0):
    """Return the selected risk-loaded tail-power bias as one float in ppm.\n\n    Nodewise velocity maxima precede cross-node population moments. The tail uses\n    only |velocity_multiplier| >= 1. Every normalized bound and the maximum retained\n    stride constraint is tested independently. Invalid inputs or no feasible candidate\n    raise ValueError.\n    """
    return 0.0
```
