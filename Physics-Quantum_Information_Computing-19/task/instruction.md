# Physics-Quantum_Information_Computing-19

## Background

Quantum key distribution derives its security from physical law rather than from computational hardness, but that guarantee is only ever as good as the description of the devices that implement it. Because practical single-photon sources remain out of reach, almost every deployed system encodes its key bits in phase-randomised weak coherent pulses, whose photon number fluctuates from pulse to pulse. An adversary is free to exploit the pulses that happen to carry more than one photon, so a practical security argument turns on estimating what share of the recorded detections can be attributed to pulses that carried exactly one. That share is never measured directly: the photon number is not recorded, and it has to be inferred from detection statistics gathered while the transmitter varies the mean photon number it aims for across a small set of settings.

The inference rests on an assumption that repays scrutiny. Intensity modulators are driven electronics, and what they deliver depends on their recent history, so the mean photon number actually emitted is pulled towards the values requested in the preceding rounds. Characterisations of fielded transmitters have measured such memory directly. Its consequence is not merely a small systematic error in the emitted mean photon number; it is a side channel, because an adversary who watches the intensities of later rounds learns something about the setting chosen earlier and may condition an attack on what is learned.

A security analysis must therefore either certify that this memory is negligible or bound what it concedes, and the second route calls for reasoning about how much freedom the memory grants an adversary to drive the statistics belonging to different settings apart. The optimisation problems that express that reasoning are the practical bottleneck of the whole analysis: they are not of the convenient form that the textbook treatment of varied intensities enjoys, and the way they are posed and solved sets how much of an implementation's genuine performance a proof is willing to give away.

## Problem

Decoy-state BB84 is almost always analysed as though the transmitter placed every pulse at its nominal mean photon number exactly, and independently of every other round. Real intensity modulators do neither: the mean photon number actually delivered sits somewhere in a narrow interval around the nominal setting, and where in that interval it sits depends on the settings selected in the few rounds immediately before, so the setting used in one round is partially readable from the intensities of the rounds that follow it. Once that leak is admitted the photon-number-resolved detection and error probabilities acquire a dependence on the intensity setting itself, the settings decouple in the parameter estimation, and the inferred single-photon contribution can be pushed to nothing. What restores a useful inference is the Cauchy–Schwarz family of constraints, which couples the photon-resolved quantities belonging to different settings through the residual indistinguishability of those settings; being nonlinear, that family has to be relaxed around a reference point before a linear program can certify a bound, and the placement of that reference point decides how much of the achievable rate survives.

Compute the asymptotic secret-key rate per emitted signal that the relaxed programs certify when they are anchored at the solution of the exact, unrelaxed ones.

The intensity density is constrained only by its support and by the memory span, with no functional form assumed for it. Double clicks are assigned a uniformly random bit, and the detection efficiency is the same in both bases. The phase-error estimate is the ratio of the bound on the single-photon error probability to the bound on the single-photon detection probability in the check basis, the single-photon emission probability common to the two cancelling; where that ratio reaches one half the privacy-amplification contribution is zero. Report alongside the result, as part of the supporting calculation, the rate the same analysis certifies for a transmitter that prepares its nominal intensities exactly.

**Source and protocol**

- Nominal intensity settings: signal $\mu = 0.48$, decoy $\nu = 0.10$, vacuum-like decoy $\omega = 1.0 \times 10^{-4}$
- Setting-selection probabilities: $p_\mu = 0.70$, $p_\nu = 0.18$, $p_\omega = 0.12$
- Each party selects the key basis with probability $0.90$ and the check basis with probability $0.10$
- Maximum relative deviation of the actual mean photon number from its nominal setting: $\delta_{\max} = 1.0 \times 10^{-3}$
- Memory span of the intensity correlations: $\xi = 3$ rounds
- Photon-number cut-off: $N_{\mathrm{cut}} = 10$

**Channel and detectors**

- Link length $35$ km through fibre of attenuation $0.2$ dB/km
- Efficiency of each of the two detectors: $0.65$
- Dark-count probability of each detector per gate: $7.2 \times 10^{-8}$
- Polarisation misalignment angle: $0.08$ rad

**Post-processing**

- Error-correction efficiency: $f_{\mathrm{EC}} = 1.16$
- Tolerated bit-error rate in the raw key-basis key: $E_{\mathrm{tol}} = 0.0065$

**Run and hardware**

- Pulse repetition rate $1.0$ GHz, with $4.0 \times 10^{11}$ signals emitted over the run
- Signal wavelength $1550$ nm; detector dead time $45$ ns
- Measured Pearson correlation coefficient between the intensities of consecutive pulses: $0.31$

**Numerical settings**

- Use a relative objective-change tolerance of $10^{-12}$ and at most $40$ refinement solves per parameter-estimation problem. Scale the change by the maximum of the two adjacent objective magnitudes and the smallest positive normal double. The final single-anchor certification LP and numerical work to obtain an exact-feasible candidate are additional to this refinement budget.
- Before returning a bound, require every original nonlinear and observed-rate inequality to hold within $10^{-9}$ absolute error, and require the candidate objective and its final single-anchor LP bound to differ by at most $10^{-8}$. Return the latter bound. Objective stability alone does not establish feasibility or optimality.
- Reference values are held at least $1 \times 10^{-12}$ inside the unit interval when the relaxation is taken

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_photon_number_bounds

Goal
----
Bound the emitted photon-number probabilities of a phase-randomised weak coherent pulse whose actual mean photon number is known only to lie within a bounded relative deviation of its nominal setting.

```python
def compute_photon_number_bounds(intensity: float, delta_max: float, n_cut: int) -> "tuple[np.ndarray, np.ndarray, float]":
    '''Bound the emitted photon-number probabilities over the admissible intensity interval.

    Parameters
    ----------
    intensity : float
        Nominal mean photon number of the setting. Strictly positive.
    delta_max : float
        Maximum relative deviation of the actual mean photon number from
        `intensity`, in [0, 1). The admissible interval of actual mean photon
        numbers is [intensity * (1 - delta_max), intensity * (1 + delta_max)].
    n_cut : int
        Photon-number cut-off. A non-negative integer.

    Returns
    -------
    lower : np.ndarray
        Shape (n_cut + 1,). Entry n is the smallest probability of emitting
        exactly n photons that is attained over the admissible interval.
    upper : np.ndarray
        Shape (n_cut + 1,). Entry n is the largest probability of emitting
        exactly n photons that is attained over the admissible interval.
    cut_mass : float
        The largest probability of emitting strictly more than `n_cut` photons
        that is attained over the admissible interval.

    Raises
    ------
    ValueError
        If `intensity` is not strictly positive, if `delta_max` is negative or
        is not below one, if `n_cut` is negative, or if
        intensity * (1 + delta_max) exceeds one, beyond which the returned
        bounds are no longer attained at the ends of the admissible interval.
    '''
    return lower, upper, cut_mass
```

### Step 2

02_cs_overlap_parameters

Goal
----
Compute, for one ordered pair of nominal intensity settings, the photon-number-resolved overlap parameter that quantifies how indistinguishable the two settings remain to an adversary once the finite-memory intensity drift of the source is taken into account.

```python
def compute_cs_overlap_parameters(intensity_a: float, intensity_b: float, intensities: "np.ndarray", probabilities: "np.ndarray", delta_max: float, correlation_range: int, n_cut: int) -> "np.ndarray":
    '''Compute the photon-number-resolved overlap parameter for one pair of settings.

    Parameters
    ----------
    intensity_a : float
        First nominal mean photon number of the pair. Strictly positive.
    intensity_b : float
        Second nominal mean photon number of the pair. Strictly positive.
    intensities : np.ndarray
        Shape (A,). Every nominal mean photon number the transmitter can select.
        All entries strictly positive.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each entry of `intensities`.
        Non-negative and summing to one.
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. A non-negative integer.

    Returns
    -------
    overlaps : np.ndarray
        Shape (n_cut + 1,). Entry n is the overlap parameter of the pair at
        photon number n. Every entry is strictly positive, and lies in (0, 1]
        whenever every nominal setting is at most one.

    Raises
    ------
    ValueError
        If either paired intensity is not strictly positive, if `intensities`
        is not a non-empty one-dimensional array of strictly positive values,
        if `probabilities` does not have the shape of `intensities` or is not a
        non-negative vector summing to one within 1e-9, if `delta_max` is
        negative or is not below one, if `correlation_range` is smaller than
        one, or if `n_cut` is negative.
    '''
    return overlaps
```

### Step 3

03_cs_boundary_values

Goal
----
Evaluate the closed interval into which the detection parameter belonging to one nominal intensity setting is confined, given the value of the same parameter for the paired setting and the pair's photon-number-resolved overlap parameter.

```python
def evaluate_cs_boundaries(parameter_values: "np.ndarray", overlaps: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    '''Evaluate the confinement interval endpoints implied by one paired parameter value.

    Parameters
    ----------
    parameter_values : np.ndarray
        Values of the detection or error parameter attached to the paired
        nominal setting. Every entry in [0, 1]. Scalars are accepted.
    overlaps : np.ndarray
        Overlap parameter of the pair at the matching photon number. Every
        entry in (0, 1]. Broadcasts against `parameter_values`.

    Returns
    -------
    lower : np.ndarray
        Lower endpoint of the confinement interval, of the broadcast shape.
    upper : np.ndarray
        Upper endpoint of the confinement interval, of the broadcast shape.

    Raises
    ------
    ValueError
        If any entry of `parameter_values` is outside [0, 1], if any entry of
        `overlaps` is outside (0, 1], if either input holds a non-finite value,
        or if the two inputs do not broadcast against one another.
    '''
    return lower, upper
```

### Step 4

04_cs_tangent_coefficients

Goal
----
Build the affine coefficients that replace each nonlinear confinement interval by a pair of half-spaces touching it at a chosen reference value, so that the confinement can be carried into a linear program.

```python
def build_cs_tangent_coefficients(reference_values: "np.ndarray", overlaps: "np.ndarray", tangent_floor: float = 1e-12) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    '''Build the affine relaxation of the confinement interval at a reference value.

    Parameters
    ----------
    reference_values : np.ndarray
        Reference values of the paired setting's parameter at which the
        relaxation is taken. Every entry in [0, 1]. Scalars are accepted.
    overlaps : np.ndarray
        Overlap parameter of the pair at the matching photon number. Every
        entry in (0, 1]. Broadcasts against `reference_values`.
    tangent_floor : float
        Distance from each end of the unit interval inside which a reference
        value is moved before the relaxation is taken. In (0, 0.5).

    Returns
    -------
    lower_slope : np.ndarray
        Slope of the lower half-space, of the broadcast shape.
    lower_offset : np.ndarray
        Offset of the lower half-space, of the broadcast shape.
    upper_slope : np.ndarray
        Slope of the upper half-space, of the broadcast shape.
    upper_offset : np.ndarray
        Offset of the upper half-space, of the broadcast shape.

    Raises
    ------
    ValueError
        If any entry of `reference_values` is outside [0, 1], if any entry of
        `overlaps` is outside (0, 1], if either input holds a non-finite value,
        if the two inputs do not broadcast against one another, or if
        `tangent_floor` is outside (0, 0.5).
    '''
    return lower_slope, lower_offset, upper_slope, upper_offset
```

### Step 5

05_channel_observables

Goal
----
Compute, for every nominal intensity setting, the sifted detection rate and the sifted error rate that the channel and detection model predicts for phase-randomised weak coherent pulses.

```python
def compute_channel_observables(intensities: "np.ndarray", transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Compute the model's sifted detection and error rates for each nominal setting.

    Parameters
    ----------
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive.
    transmittance : float
        Overall probability that an emitted photon reaches and fires a
        detector, combining channel loss and detector efficiency. In [0, 1].
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).

    Returns
    -------
    key_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate in the basis used to distil the key.
    check_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate in the basis used to estimate the
        phase error.
    check_basis_error_rates : np.ndarray
        Shape (A,). Sifted error rate in the basis used to estimate the phase
        error.

    Raises
    ------
    ValueError
        If `intensities` is not a non-empty one-dimensional array of strictly
        positive values, if `transmittance` is outside [0, 1], if
        `misalignment` is outside [0, pi / 4], or if `dark_count` is negative
        or is not below one.
    '''
    return key_basis_rates, check_basis_rates, check_basis_error_rates
```

### Step 6

06_fock_reference_points

Goal
----
Compute, for every photon number up to the cut-off, the conditional detection probability and the conditional error probability that the channel and detection model assigns to a signal carrying exactly that many photons.

```python
def compute_fock_reference_points(n_cut: int, transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray]":
    '''Compute the model's Fock-resolved detection and error probabilities.

    Parameters
    ----------
    n_cut : int
        Photon-number cut-off. A non-negative integer.
    transmittance : float
        Overall probability that an emitted photon reaches and fires a
        detector, combining channel loss and detector efficiency. In [0, 1].
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).

    Returns
    -------
    yield_reference : np.ndarray
        Shape (n_cut + 1,). Entry n is the conditional detection probability of
        a signal carrying exactly n photons.
    error_reference : np.ndarray
        Shape (n_cut + 1,). Entry n is the conditional error probability of a
        signal carrying exactly n photons in the basis used to estimate the
        phase error, averaged over the encoded bit.

    Raises
    ------
    ValueError
        If `n_cut` is negative, if `transmittance` is outside [0, 1], if
        `misalignment` is outside [0, pi / 4], or if `dark_count` is negative
        or is not below one.
    '''
    return yield_reference, error_reference
```

### Step 7

07_outer_linearised_program

Goal
----
Solve one linear program that bounds the single-photon detection or error parameter of the signal setting, using the sifted rates together with the affine relaxation of the confinement constraints taken at a supplied collection of reference points.

```python
def solve_outer_linearised_program(
    observed_rates: 'np.ndarray',
    reference_points: 'np.ndarray',
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    maximise: bool,
) -> 'tuple[float, np.ndarray]':
    '''Bound the signal single-photon parameter with one linear program.

    Parameters
    ----------
    observed_rates : np.ndarray
        Shape (A,). Sifted rate for each nominal setting, conditioned on
        matching bases and that setting. Every entry in [0, 1].
    reference_points : np.ndarray
        Shape (R, A, n_cut + 1) with R >= 1. Relaxation points; entry
        (r, a, n) is the reference value used for setting a at photon number n
        in the r-th point. Every entry in [0, 1].
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive. Entry
        zero is the signal setting.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each nominal setting.
        Non-negative and summing to one.
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. At least one.
    maximise : bool
        When False the signal setting's single-photon parameter is minimised,
        giving a lower bound; when True it is maximised, giving an upper bound.
        Required.

    Returns
    -------
    objective : float
        Optimal value of the signal setting's single-photon parameter.
    parameters : np.ndarray
        Shape (A, n_cut + 1). An optimal assignment of the photon-resolved
        parameters attaining `objective`.

    Notes
    -----
    Resolve the objective and affine constraints accurately enough for
    absolute errors below 1e-9. For HiGHS, primal and dual feasibility
    tolerances of 1e-10 with a common factor of 1000 applied to the
    objective and inequalities are sufficient for these instances.
    Undo the objective scaling before returning. Equivalent accurate
    LP implementations are accepted; the scaling changes no constraint.

    Raises
    ------
    ValueError
        If any array argument has the wrong shape or holds values outside its
        stated range, if `n_cut` is smaller than one, if `correlation_range` is
        smaller than one, if `delta_max` is negative or is not below one, or if
        the linear program is not solved to optimality.
    '''
    return objective, parameters
```

### Step 8

08_certified_parameter_bounds

Goal
----
Resolve the exact Cauchy-Schwarz-constrained parameter-estimation problems, then return conservative bounds from one final LP per problem, anchored only at its numerically feasible solution.

```python
def compute_certified_parameter_bounds(
    key_basis_rates: 'np.ndarray',
    check_basis_rates: 'np.ndarray',
    check_basis_error_rates: 'np.ndarray',
    yield_reference: 'np.ndarray',
    error_reference: 'np.ndarray',
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> 'tuple[float, float, float, float, float, float]':
    '''Certify bounds on the signal setting's single-photon parameters.

    Parameters
    ----------
    key_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate per setting in the key basis.
    check_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate per setting in the check basis.
    check_basis_error_rates : np.ndarray
        Shape (A,). Sifted error rate per setting in the check basis.
    yield_reference : np.ndarray
        Shape (n_cut + 1,). Starting reference values for the two detection
        programs, indexed by photon number. Every entry in [0, 1].
    error_reference : np.ndarray
        Shape (n_cut + 1,). Starting reference values for the error program,
        indexed by photon number. Every entry in [0, 1].
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive. Entry
        zero is the signal setting.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each nominal setting.
        Non-negative and summing to one.
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. At least one.
    max_iterations : int
        Positive refinement budget for each parameter-estimation problem.
        For the accumulated-tangent method, count its initial LP and each
        subsequent refinement LP, up to this limit. The final certifying
        LP and numerical work to obtain an exact-feasible candidate are
        additional. A value of one deliberately returns the initial
        channel-reference LP bound without claiming exact optimality.
    objective_rtol : float
        Strictly positive relative stopping tolerance. Compare adjacent
        refinement objectives v_old and v_new using
        abs(v_new-v_old) <= objective_rtol *
        max(abs(v_new), abs(v_old), np.finfo(float).tiny).
        For a budget above one, also require all original nonlinear and
        observed-rate inequalities within absolute 1e-9 and the gap
        between the candidate objective and its single-anchor LP bound
        at most 1e-8. Objective stability alone is insufficient.

    Returns
    -------
    key_yield_bound : float
        Lower bound from the final single-anchor LP in the key basis.
        Return this LP value, not the accumulated-refinement objective.
    check_yield_bound : float
        Lower bound on the same parameter in the check basis.
    error_bound : float
        Upper bound on the signal setting's single-photon error parameter in
        the check basis.
    key_yield_residual : float
        Absolute difference between the exact-feasible candidate's
        objective and the returned single-anchor LP bound. In one-pass
        mode, use the supplied reference's signal one-photon value;
        that residual is diagnostic and does not certify optimality.
    check_yield_residual : float
        The same residual for `check_yield_bound`.
    error_residual : float
        The same residual for `error_bound`.

    Raises
    ------
    ValueError
        If any array argument has the wrong shape or holds values outside its
        stated range, if `max_iterations` is smaller than one, if
        `objective_rtol` is not strictly positive, any LP is unsuccessful,
        or the budget above one is exhausted without meeting the relative,
        original-feasibility and final-gap conditions.
    '''
    return (key_yield_bound, check_yield_bound, error_bound,
            key_yield_residual, check_yield_residual, error_residual)
```

### Step 9

09_certified_key_rate

Goal
----
Compute the certified asymptotic secret-key rate per emitted signal of a decoy-state protocol whose transmitter prepares its nominal intensities only to within a bounded relative deviation, with the deviations correlated across a finite span of rounds.

```python
def compute_certified_key_rate(
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    key_basis_probability: float,
    distance_km: float,
    attenuation_db_per_km: float,
    detector_efficiency: float,
    dark_count: float,
    misalignment: float,
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    error_correction_efficiency: float,
    tolerated_error_rate: float,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> float:
    '''Compute the certified asymptotic secret-key rate per emitted signal.

    Parameters
    ----------
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive. Entry
        zero is the signal setting.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each nominal setting.
        Non-negative and summing to one.
    key_basis_probability : float
        Probability with which each party selects the key basis, in [0, 1].
    distance_km : float
        Link length in kilometres. Non-negative.
    attenuation_db_per_km : float
        Fibre attenuation coefficient in decibels per kilometre. Non-negative.
    detector_efficiency : float
        Efficiency of each detector, in [0, 1].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. At least one.
    error_correction_efficiency : float
        Error-correction efficiency factor. At least one.
    tolerated_error_rate : float
        Bit-error rate the protocol is designed to tolerate in the raw
        key-basis key, in [0, 0.5].
    max_iterations : int
        Positive refinement budget passed to Step 08. One deliberately
        uses the unrefined channel-reference bounds. For larger budgets,
        the final single-anchor certification LP and exact-feasibility
        polishing are additional to the refinement solves.
    objective_rtol : float
        Strictly positive relative refinement tolerance, as in Step 08.
        Its original-feasibility and final-gap checks must also pass.

    Returns
    -------
    key_rate : float
        Certified asymptotic secret-key rate per emitted signal. May be
        non-positive, in which case the configuration supports no key.

    Raises
    ------
    ValueError
        If any argument lies outside its stated range or any array argument has
        the wrong shape, any LP fails, or Step 08 exhausts a budget above
        one without satisfying its convergence and certification checks.
    '''
    return key_rate
```
