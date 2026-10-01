# Chemistry-Computational_Chemistry-30

## Problem

Assess whether a calculated triplet energy gap is compatible with a long phosphorescence tail using the study of simultaneous prompt fluorescence, thermally activated delayed fluorescence and room-temperature phosphorescence in 1.8-mDTAZ-PhtCz and its supplementary information. Retrieve the comprehensive three-excited-state equations and two rate-ratio relations in Note S2, Eqs. S5–S9; the Note S2 values of τ_S1, k_ISC, k_rISC2, k_BT and estimated k_IC; the phosphorescence lifetime discussed beside Fig. S26; and the ROKS T₂ and T₁ energies in Table S9. Treat these printed quantities as exact nominal inputs, use the supplementary parameter set rather than main-text Table 2, and interpret the retrieved phosphorescence lifetime as the asymptotic decay time of T₁ population rather than its intrinsic lifetime or a fit to raw burst-mode data.

Set all annihilation coefficients to zero, initialize normalized populations S(0)=1 and U(0)=L(0)=0 for S₁, T₂ and T₁, and retain exactly the linear transitions in Eqs. S5–S7. For free parameters Δ=E(T₂)−E(T₁)≥0 and α≥0, multiply the right-hand side of each retrieved rate-ratio relation in Eqs. S8–S9 by the same dimensionless α, evaluating both at Δ and the retrieved k_BT, so α=1 preserves the source relations. Replace the intrinsic T₁ ground-state loss 1/τ_T1 by d=κ+k_nr,T1 with an independently known hypothetical radiative rate κ=1.00 s⁻¹, unknown k_nr,T1≥0 and measured signal I_P(t)=κL(t); keep all other retrieved rates fixed and time-independent, with no continued excitation, extra state or additional loss channel.

Determine the complete physically admissible set of (Δ,α,d) reproducing the target asymptotic lifetime, justify that each matching mode is the observable slowest decay, and obtain the minimum Δ allowed when α=1. At the retrieved ROKS gap, decide whether α=1 is admissible and determine the global maximum α, distinguishing what this implies for the calculated gap itself from what it implies for the combined gap-and-rate-ratio hypothesis. Assess whether complete calibrated population traces at this one temperature can separately identify Δ and α, and specify an additional measurement and the assumptions needed to separate them. Return the maximum α at the ROKS gap as the final numerical target with approximately 0.5% relative accuracy; equivalent notation and analytic or numerical methods are accepted, but a parameter scan alone does not establish a global bound, and neither α nor κ should be presented as experimentally identified by this idealized calculation.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_source_inputs.py

Goal
----
Retrieve the comprehensive model and nominal parameters from Supplementary Note S2, including the estimated IC rate and the lifetime beside Fig. S26. Retrieve the ROKS T₂ and T₁ energies from Table S9. Return the values in consistent SI units and the triplet gap in eV.

```python
import numpy as np

def prepare_inputs(source: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """Convert the retrieved source quantities to the kinetic benchmark.

    Parameters
    ----------
    source : array-like, shape (8,)
        [tau_S1_ns, k_ISC, k_rISC2, k_IC, kBT_meV,
         tail_lifetime_ms, E_T2_eV, E_T1_eV]. Rates are in s^-1.
        Retrieve the Note S2 / Fig. S26 / Table S9 quantities, not
        main-text Table 2. Printed approximate values are nominal inputs.
    kappa : float
        Independently known T1 radiative rate in s^-1, strictly positive.

    Returns
    -------
    np.ndarray
        Shape (8,): [tau_s, i, u, c, theta, lambda, gap, kappa],
        in seconds, s^-1, and eV. lambda is the inverse tail lifetime
        and gap is E_T2-E_T1. Do not identify lambda with intrinsic d.

    Raises
    ------
    ValueError
        If input is not real numeric and finite, source has the wrong
        shape, any of its first six values or kappa is nonpositive,
        E_T2<E_T1, lambda<=kappa, or converted results are nonfinite.
    """
    return np.empty(8, dtype=float)  # placeholder
```

### Step 2

02_population_generator.py

Goal
----
Construct and return the linear population generator in the order S₁, T₂, T₁, with z=α exp(−Δ/θ) and d=κ+k_nr,T1. Include both T₁ escape pathways. Its column sums must reproduce the total-population balance (S+U+L)′=−S/τ_S1−dL.

```python
import numpy as np

def population_generator(tau_s: float, i: float, u: float, c: float,
                         z: float, d: float) -> np.ndarray:
    """Build M for x'=Mx, x=(S1,T2,T1), without annihilation.

    tau_s is intrinsic singlet lifetime (s); i,u,c are positive ISC,
    upper-triplet RISC and IC rates (s^-1). z>=0 is the dimensionless
    shared factor alpha*exp(-gap/theta). T1->S1 has rate u*z,
    T1->T2 has rate c*z, and d>=0 is total T1 ground-state loss.
    S1 ground-state loss is 1/tau_s. No other transitions occur.

    Returns
    -------
    np.ndarray
        Shape (3,3), finite float column-vector generator. Its column
        sums encode ground-state loss and internal transfer conserves
        total excitation population.

    Raises
    ------
    ValueError
        If an input is not a finite real scalar, tau_s,i,u,c are not
        positive, z or d is negative, or matrix entries overflow.
    """
    return np.empty((3, 3), dtype=float)  # placeholder
```

### Step 3

03_exact_tail_constraint.py

Goal
----
Derive the relation between d and z required by the target slow decay rate λ. Eliminate the singlet and upper-triplet amplitudes at the target eigenvalue and evaluate the resulting coefficients.

```python
import numpy as np

def tail_coefficients(tau_s: float, i: float, u: float, c: float,
                      decay: float) -> np.ndarray:
    """Eliminate S1 and T2 at a specified exp(-decay*t) eigenmode.

    tau_s is positive intrinsic singlet lifetime in seconds; i,u,c are
    positive ISC, RISC2, and IC rates in s^-1; decay>0 is the measured
    inverse tail lifetime, not an intrinsic loss. In the model of step 2
    write a=1/tau_s+i, b=u+c, B=(a-decay)*(b-decay)-i*u.
    Determine the coefficient H such that the exact eigenvalue equation
    for the T1 loss becomes d=decay-H*z. Do not set decay to zero in
    the fast-state elimination.

    Returns
    -------
    np.ndarray
        Shape (3,), [a, B, H] in s^-1, s^-2, s^-1.

    Raises
    ------
    ValueError
        If inputs are not finite real positive scalars, a<=decay,
        b<=decay, B<=0, H<=0, or any computed coefficient is nonfinite.
        This contract restricts the target to below the fast-block poles.
    """
    return np.empty(3, dtype=float)  # placeholder
```

### Step 4

04_admissible_parameter_set.py

Goal
----
Determine the entire set of nonnegative Δ, α and nonnegative triplet nonradiative loss consistent with the lifetime. Give both the necessary inequality and a constructive expression for d.

```python
import numpy as np

def admissible_point(H: float, decay: float, kappa: float, theta: float,
                     gap: float, alpha: float) -> np.ndarray:
    """Evaluate a member of the exact lifetime-matching parameter family.

    H>0 is from tail_coefficients; decay>=kappa>0 are s^-1;
    theta>0 and gap>=0 are eV; alpha>=0 is dimensionless.
    Calculate z=alpha*exp(-gap/theta), d=decay-H*z and the maximum
    allowed z from d>=kappa. A negative required d is diagnostic output,
    not an exception. The flag is 1 only for an admissible d.
    To handle arithmetic at the boundary, snap d to kappa when their
    difference is at most 64*machine_epsilon*max(1,decay,kappa).
    This numerical guard is not an experimental uncertainty allowance.

    Returns
    -------
    np.ndarray
        Shape (4,), [z,d,z_max,flag], with flag equal to 0.0 or 1.0.
        Together with gap>=0, alpha>=0 and d=decay-H*z, z<=z_max
        characterizes the whole physically admissible family.

    Raises
    ------
    ValueError
        If inputs are not finite real scalars, H,theta,kappa are
        nonpositive, decay<kappa, gap or alpha is negative, or
        calculated outputs are nonfinite.
    """
    return np.empty(4, dtype=float)  # placeholder
```

### Step 5

05_slow_mode_certificate.py

Goal
----
Establish whether the dominant kinetic eigenmode is observable in T₁ after S₁-only excitation. Implement slow_mode to return its decay rate and spectral-projection amplitude, including the zero-back-transfer and disconnected limits. The complete scientific reasoning must justify why this mode is the asymptotic physical tail.

```python
import numpy as np

def slow_mode(matrix: np.ndarray) -> np.ndarray:
    """Compute the asymptotic T1 contribution of the dominant kinetic mode.

    matrix is a real (3,3) column-vector population generator in s^-1,
    ordered S1,T2,T1. It has nonnegative off-diagonal rates, column sums
    <=0, a stable simple real dominant eigenvalue, and a nonsingular
    eigenvector basis. For x(0)=(1,0,0), return the decay rate and the
    coefficient of its exponential in L(t). The amplitude can be zero
    in a disconnected model: then this mode is not observed in T1.
    Use the spectral projection; do not assume every eigenvalue appears
    with nonzero amplitude. At z=0 the fed lower triplet still has a tail.

    Returns
    -------
    np.ndarray
        Shape (2,), [-dominant_eigenvalue, T1_tail_amplitude].

    Raises
    ------
    ValueError
        If matrix is not finite real shape (3,3), an off-diagonal entry
        is negative, a column sum exceeds 64*eps*max(1,max(abs(matrix))),
        the dominant eigenvalue is nonnegative, nonreal or not simple,
        the eigenvector basis is singular, or results are nonfinite.
        Numerical realness/simplicity uses 1e-10*max(1,abs(dominant)).
        Imaginary amplitude above 1e-10*max(1,abs(real(amplitude))) is invalid.
    """
    return np.empty(2, dtype=float)  # placeholder
```

### Step 6

06_roks_prefactor_bound.py

Goal
----
Implement kinetic_bounds to return the maximal admissible Boltzmann-prefactor product, the global minimum nonnegative gap for α=1, and the global maximum α at a supplied gap. Apply the result to the retrieved ROKS gap, identify the nonradiative-loss boundary, and assess the conditional validity of α=1.

```python
import numpy as np

def kinetic_bounds(H: float, decay: float, kappa: float,
                   theta: float, gap: float) -> np.ndarray:
    """Compute global kinetic bounds from d=decay-H*z and d>=kappa.

    H>0, decay>kappa>0 are s^-1; theta>0 and gap>=0 are eV.
    The gap argument is the retrieved ROKS T2-T1 gap when applying this
    step to the paper. z=alpha*exp(-gap/theta). Return its maximal value,
    the smallest nonnegative gap compatible with alpha=1, and the
    maximal alpha at the supplied gap. Bounds are attained at d=kappa,
    except that the minimum gap is zero if alpha=1 already fits there.
    Do not infer an unconditional measured energy gap from this result.

    Returns
    -------
    np.ndarray
        Shape (3,), [z_max, minimum_gap, maximum_alpha].

    Raises
    ------
    ValueError
        If inputs are not finite real scalars, H,theta,kappa are
        nonpositive, decay<=kappa, gap<0, or bounds are nonfinite or
        z_max is nonpositive in floating-point arithmetic.
    """
    return np.empty(3, dtype=float)  # placeholder
```

### Step 7

07_identifiability_experiment.py

Goal
----
Explain why one-temperature trajectories cannot separate the energy gap and common multiplier, then implement thermal_identification for the proposed additional measurement: recover the gap and multiplier from two calibrated kinetic factors at distinct temperatures, assuming both parameters remain constant. Reject zero factors, which carry no finite identifiable gap under this protocol.

```python
import numpy as np

def thermal_identification(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Separate gap and prefactor using two calibrated kinetic factors.

    theta and z are real arrays of shape (2,). theta contains distinct
    positive kBT values in eV; z contains strictly positive measured
    factors k_rISC1/k_rISC2 (=k_rIC/k_IC under this model).
    Assume the same nonnegative gap and positive alpha at both
    temperatures, with other rates independently measured. Infer them
    from log(z)=log(alpha)-gap/theta. Either temperature order is valid.
    A single temperature, or alpha=0 at both temperatures, cannot
    identify the gap; zero factors are therefore outside this contract.

    Returns
    -------
    np.ndarray
        Shape (2,), [gap_eV,alpha], gap>=0 and alpha>0.

    Raises
    ------
    ValueError
        If inputs are not finite real arrays of shape (2,), a theta or
        z value is nonpositive, the two theta values are equal, the
        inferred gap is negative, or results are nonfinite or alpha
        is zero after floating-point underflow.
    """
    return np.empty(2, dtype=float)  # placeholder
```

### Step 8

08_final_orchestrator.py

Goal
----
Combine source retrieval, exact tail constraint, physical bounds and computed gap to report the globally maximal dimensionless α. Include the admissible family, α=1 gap bound, mode-validity argument and conditional interpretation.

```python
import numpy as np
from typing import Optional

def solve_triplet_bound(source: Optional[np.ndarray] = None,
                        kappa: float = 1.0) -> float:
    """Return the maximal common escape multiplier at the computed gap.

    If source is None, retrieve the nominal Note S2, Fig. S26 and Table
    S9 inputs from Dou et al., DOI 10.1038/s41377-025-02063-x.
    Otherwise source uses the eight-value source-unit layout documented
    by prepare_inputs. kappa is the known positive T1 radiative rate.
    Chain the previous computational steps: unit conversion, exact
    lifetime elimination, global bounds, admissible boundary, generator
    and dominant-mode calculation. Use public step functions here;
    reference/oracle implementations chain only oracle functions.
    The two-temperature step is a separate proposed experiment, not
    extra input required for this one-temperature bound.

    Returns
    -------
    float
        Maximal alpha at the supplied or retrieved ROKS gap, attained at
        zero T1 nonradiative loss. This is a bound, not an identified gap.

    Raises
    ------
    ValueError
        For nonreal/nonfinite inputs, wrong source shape, nonpositive
        source lifetimes/rates/thermal energy or kappa, inverted triplet
        energy order, decay<=kappa, nonfinite converted values, a target
        outside the positive fast-block elimination domain (a<=decay,
        u+c<=decay, B<=0 or H<=0), nonfinite bounds/matrix/spectral
        results, a nonphysical generator, a nonstable/nonsimple/nonreal
        dominant pole or singular eigenvector basis. Also raise if the
        constructed boundary fails admissible_point's rounding guard,
        its dominant decay differs from the target by more than 1e-6
        relative, or its T1 tail amplitude is nonpositive. The matrix
        and spectral numerical conventions are those of slow_mode.
    """
    return 0.0  # placeholder
```
