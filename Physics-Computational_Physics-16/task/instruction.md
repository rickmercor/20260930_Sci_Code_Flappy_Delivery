# Physics-Computational_Physics-16

## Background

Neutron populations fluctuate because individual captures, fissions and emissions occur at random. Delayed-neutron precursors store part of this branching history and release neutrons on longer time scales. Population averages describe only part of that behavior; uncertainty also depends on correlations between the particles involved in shared events.

In circulating-fuel reactors, the fuel transports precursors between irradiated and external regions. Their location at decay affects the reactor's neutron supply. Modeling the resulting fluctuations matters when comparing approximate descriptions of the reactor's dynamics.

## Problem

Low-population fluctuations in circulating-fuel reactors couple neutron multiplication to the motion and decay of delayed-neutron precursors. A recent perfectly mixed, two-region model describes this process both through discrete events and through a continuous Itô approximation, and compares them in a ramp-up benchmark.

Use the benchmark configuration: prompt neutron generation time Λ = 10⁻³ s, constant external source S = 8800 s⁻¹, total delayed fraction β = 0.0065, six precursor groups with relative fractions α = (0.033, 0.219, 0.196, 0.395, 0.115, 0.042) and decay constants λ = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01) s⁻¹, prompt-neutron multiplicity probabilities (0.027, 0.158, 0.339, 0.305, 0.133, 0.038) for yields 0 to 5, zero initial populations and covariances, reactivity ρ(t) = −0.01 + 0.016·min(t/10, 1), and residence times τ_c(t) = 1000 − 990·min(t/5, 1) s and τ_e(t) = 1000 − 985·min(t/5, 1) s, with t in seconds. Calculate the percentage by which the continuous Itô model underestimates the variance of the **ex-core group-1 precursor population at t = 8 s**, relative to the discrete-event model. Use ensemble variances of the continuous-time models, with the Itô stochastic terms as specified in the source. For the discrete-event model, use the benchmark's tabulated prompt-neutron multiplicity distribution, independent of a Poisson total delayed yield whose mean is consistent with the benchmark delayed fraction; allocate that delayed yield multinomially using the relative group fractions.

The required percentage is 100 times the discrete-event variance minus the Itô variance, divided by the discrete-event variance. Explain how the models' stochastic terms enter the variance calculation, and report the percentage rounded to **five significant figures**.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

compute_fission_moments

Goal
----
Implement compute_fission_moments which computes the first and raw second moments of a fission population increment.

```python
def compute_fission_moments(probabilities: "np.ndarray", beta: float, alpha: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return moments of the increment in neutron and six core-precursor populations.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Shape (K,), probabilities of prompt yields 0 through K-1; sum one
        and have positive mean. The prompt yield is independent of delayed yields.
    beta : float
        Total delayed fraction, 0 <= beta < 1. Set the delayed-yield mean
        by consistency with the source model's mean precursor-production rate.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
        The Poisson total delayed yield is split multinomially with these weights.
    
    Returns
    -------
    mean_increment : np.ndarray
        Shape (7,), ordered neutron increment then groups 1 through 6.
    raw_second : np.ndarray
        Shape (7, 7), E[delta X delta X^T], including neutron consumption.
        This is a raw second moment, before subtracting the mean outer product.
    """
    return result
```

### Step 2

build_mean_drift

Goal
----
Implement build_mean_drift which constructs the linear mean-population operator for two perfectly mixed fuel regions.

```python
def build_mean_drift(rho: float, tau_core: float, tau_excore: float, generation_time: float, beta: float, alpha: "np.ndarray", decay: "np.ndarray") -> "np.ndarray":
    """Construct the homogeneous drift operator of the source's six-group model.
    
    Parameters
    ----------
    rho : float
        Reactivity at the current time.
    tau_core, tau_excore : float
        Positive residence times in seconds; positive infinity disables that transfer.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    alpha : np.ndarray
        Shape (6,), nonnegative relative group fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative group decay constants in inverse seconds.
    
    Returns
    -------
    drift : np.ndarray
        Shape (13, 13), A such that dm/dt = A m + S e_n.
        State order is neutron, core groups 1-6, ex-core groups 1-6;
        e_n selects the neutron coordinate. The external source is separate.
    """
    return result
```

### Step 3

build_jump_noise

Goal
----
Implement build_jump_noise which computes the instantaneous covariance-production matrix of the discrete-event model.

```python
def build_jump_noise(mean: "np.ndarray", phi: float, gamma: float, source: float, tau_core: float, tau_excore: float, decay: "np.ndarray", fission_second: "np.ndarray") -> "np.ndarray":
    """Compute the noise term Q in dC/dt = A C + C A^T + Q for the jump model.
    
    Parameters
    ----------
    mean : np.ndarray
        Shape (13,), nonnegative expected populations in neutron/core/ex-core order.
    phi, gamma : float
        Nonnegative per-neutron fission and loss rates in inverse seconds.
    source : float
        Nonnegative Poisson neutron-source intensity in inverse seconds.
    tau_core, tau_excore : float
        Positive residence times in seconds; positive infinity disables that transfer.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants in inverse seconds.
    fission_second : np.ndarray
        Shape (7, 7), raw second moment of a fission increment, ordered
        neutron followed by the six core groups, including neutron consumption.
    
    Returns
    -------
    noise : np.ndarray
        Shape (13, 13), expected instantaneous quadratic increment per second,
        in neutron, core groups 1-6, ex-core groups 1-6 order.
    """
    return result
```

### Step 4

build_sde_noise

Goal
----
Implement build_sde_noise which computes covariance production for the source paper’s continuous stochastic differential equation model.

```python
def build_sde_noise(mean: "np.ndarray", phi: float, gamma: float, prompt_increment_second: float) -> "np.ndarray":
    """Return the noise matrix of the paper's continuous Itô approximation.
    
    Parameters
    ----------
    mean : np.ndarray
        Shape (13,), nonnegative expected populations, ordered neutron,
        core groups 1-6, ex-core groups 1-6. Use the nonnegative-state regime
        guaranteed by the benchmark's source-positivity condition.
    phi, gamma : float
        Nonnegative per-neutron fission and loss rates in inverse seconds.
    prompt_increment_second : float
        E[(nu_p - 1)^2], the raw second moment of the prompt neutron increment.
    
    Returns
    -------
    noise : np.ndarray
        Shape (13, 13), Q in the continuous model's covariance equation.
        Use the stochastic terms specified in the source's Itô model;
        finite-time-step discretization effects are outside this function.
    """
    return result
```

### Step 5

evaluate_moment_rhs

Goal
----
Implement evaluate_moment_rhs which evaluates the coupled mean and covariance derivatives for both reactor models during a ramp transient.

```python
def evaluate_moment_rhs(t: float, state: "np.ndarray", fission_mean: "np.ndarray", fission_second: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "np.ndarray":
    """Evaluate a shared mean and two covariance derivatives under linear ramp schedules.

    Parameters
    ----------
    t : float
        Nonnegative time in seconds.
    state : np.ndarray
        Shape (351,): mean (13,), jump covariance flattened row-major (169,),
        then SDE covariance flattened row-major (169,). The state coordinate
        order is neutron, core groups 1-6, ex-core groups 1-6.
    fission_mean : np.ndarray
        Shape (7,), mean fission increment in neutron/core coordinates.
    fission_second : np.ndarray
        Shape (7, 7), raw second moment of that increment.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Nonnegative, constant external neutron-source intensity per second.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants per second.
    rho_schedule, tau_core_schedule, tau_excore_schedule : np.ndarray
        Shape (3,) each, (x_0, x_1, T) for the reactivity and for the core and
        ex-core residence times in seconds: the quantity follows the linear
        ramp x(t) = x_0 + (x_1 - x_0) * min(t / T, 1) with ramp duration T > 0,
        so it stays at x_1 after t = T. Residence-time values are positive.

    Returns
    -------
    derivative : np.ndarray
        Shape (351,), derivatives in the same layout as state. Covariances
        are central second moments. Model parameters are in the source's
        nonnegative SDE regime and give nonnegative event rates.
    """
    return result
```

### Step 6

propagate_moments

Goal
----
Implement propagate_moments which evolves the ensemble mean and both population covariance matrices from an empty reactor.

```python
def propagate_moments(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Propagate the continuous model moments with zero initial populations and covariance.

    Parameters
    ----------
    t_end : float
        Nonnegative observation time in seconds.
    probabilities : np.ndarray
        Shape (K,), prompt-yield probabilities for 0 through K-1, with positive mean.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Constant source intensity per second, satisfying the source model's
        positivity condition for the continuous SDE over the interval.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants per second.
    rho_schedule, tau_core_schedule, tau_excore_schedule : np.ndarray
        Shape (3,) each, (x_0, x_1, T) for the reactivity and for the core and
        ex-core residence times in seconds: the quantity follows the linear
        ramp x(t) = x_0 + (x_1 - x_0) * min(t / T, 1) with ramp duration T > 0,
        so it stays at x_1 after t = T. Residence-time values are positive.

    Returns
    -------
    covariance_jump : np.ndarray
        Shape (13, 13), central covariance for the discrete-event process.
    covariance_sde : np.ndarray
        Shape (13, 13), central covariance for the continuous Itô approximation.
        All outputs use neutron, core groups 1-6, ex-core groups 1-6 order.
        Resolve the continuous moments to relative accuracy 1e-7 with
        absolute accuracy 1e-8 for components near zero.

    Raises
    ------
    ValueError
        If t_end is negative.
    """
    return result
```

### Step 7

calculate_variance_shortfall

Goal
----
Implement calculate_variance_shortfall which compares the ex-core precursor variances predicted by the two benchmark models.

```python
def calculate_variance_shortfall(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray", group: int = 1) -> float:
    """Run the complete pipeline and report the SDE variance shortfall.

    Parameters
    ----------
    t_end : float
        Positive observation time in seconds.
    probabilities : np.ndarray
        Shape (K,), prompt-yield probabilities for yields 0 through K-1.
        The prompt yield is independent of the Poisson total delayed yield,
        which is split among groups by the relative fractions alpha.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Constant external neutron-source intensity per second, satisfying
        the source model's positivity condition over the interval.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants per second.
    rho_schedule, tau_core_schedule, tau_excore_schedule : np.ndarray
        Shape (3,) each, (x_0, x_1, T) for the reactivity and for the core and
        ex-core residence times in seconds: the quantity follows the linear
        ramp x(t) = x_0 + (x_1 - x_0) * min(t / T, 1) with ramp duration T > 0,
        so it stays at x_1 after t = T. Residence-time values are positive.
    group : int
        Delayed-neutron precursor group, numbered 1 through 6.
        Defaults to 1, the benchmark's ex-core group.
        Initially all populations and covariances are zero.

    Returns
    -------
    shortfall : float
        100 * (V_jump - V_sde) / V_jump for the selected ex-core group.
        Return the unrounded percentage for the continuous model statistics.
        Zero initial conditions and a constant source make both variances
        proportional to source, leaving this percentage invariant.
    """
    return result
```
