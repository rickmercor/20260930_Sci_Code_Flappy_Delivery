# Biology-Ecology-48

## Background

Ultra-high-frequency animal tracks can reveal real changes in heading rather than changes created by a fixed sampling interval. Turning events occur at uneven times, so each straight segment carries a duration as well as a travel speed and turning angle. Hidden behavioural modes can govern those movement measurements while distance to a central place and time of day affect switching between modes. Computing a marginal likelihood requires accounting for every possible state sequence, not merely the most likely path. Sensitivity of that marginal likelihood to both an emission parameter and a transition parameter can reveal coupling caused by uncertain behavioural states.

## Problem

High-frequency animal tracks can be segmented at real turns, yielding steps with unequal durations. An event-indexed hidden Markov model can use these movement decisions to infer behavioural modes. For the supplied turning points, compute the mixed derivative of the **marginal log likelihood** at fixed parameters:

$$
\frac{\partial^2\log L}
{\partial\kappa_1\,\partial a^{(12)}_2}.
$$

Here, \(\kappa_1=\texttt{kappa[0]}\) is state 1’s carved turning-angle concentration, and \(a^{(12)}_2=\texttt{a[0][2]}\) is the cosine-of-clock-hour coefficient for the transition from state 1 to state 2. Use the source’s carved von Mises turning-angle density, conditionally independent gamma **duration and speed** densities, and reference-category transition logits with distance and cyclic time-of-day covariates. Transitions occur at turns; do not raise a transition matrix to an elapsed-time power.

Number the four steps and their arrival events \(t=0,1,2,3\). Let \(P_t\) denote `points[t]`, and define the current displacement \(v_t=P_{t+1}-P_t\). The incoming direction is \(u_0=h\) for the first step and \(u_t=v_{t-1}\) for subsequent steps. Compute the signed turning angle as

$$
\phi_t=
\operatorname{wrap}_{[-\pi,\pi)}
\left[
\operatorname{atan2}
\left(
u_{t,x}v_{t,y}-u_{t,y}v_{t,x},
u_t\cdot v_t
\right)
\right].
$$

The duration is \(\tau_t=\texttt{times\_hours[t+1]}-\texttt{times\_hours[t]}\), the speed is \(\lVert v_t\rVert/\tau_t\), the distance covariate is measured from the arrival point \(P_{t+1}\) to `centre`, and the clock-hour covariate \(H_t\) is `times_hours[t+1]` modulo 24.

In each two-state transition row, the self-transition predictor is zero. The predictor for the other state is

$$
a^{(ij)}_0+a^{(ij)}_1d_t
+a^{(ij)}_2\cos\!\left(\frac{\pi H_t}{12}\right)
+a^{(ij)}_3\sin\!\left(\frac{\pi H_t}{12}\right).
$$

For \(t=1,2,3\), \(\Gamma_t\) maps the state at event \(t-1\) to the state at event \(t\) using **event \(t\)’s arrival covariates**. The fixed initial distribution \(\delta\) applies directly to the emission at event 0, with no preceding transition, and is independent of both differentiated parameters.

In your reasoning, report the event durations, speeds, angles and clock hours; the two carved-density normalizing constants; the state-1-to-state-2 transition probability at the second event (\(t=1\)); the marginal log likelihood; and the requested mixed derivative. Treat the observation as \((\phi,\tau,s)\), where \(s\) is **speed**, so no length-to-speed Jacobian enters. Use these exact inputs:

```python
points = [[0., 0.], [.5, .1], [1., .7], [1.4, 1.], [2.2, .3]]
times_hours = [22.8, 23.1, 23.55, 24.1, 24.6]

h = [1., 0.]
centre = [.5, .1]

kappa = [2.4, 5.7]
width = [1.3, .42]

time_shape = [2.2, 4.1]
time_rate = [3.2, 4.7]
speed_shape = [3.1, 5.3]
speed_rate = [2.1, 3.7]

# Rows correspond to transitions 1→2 and 2→1, respectively.
# Columns are intercept, distance, cosine, and sine coefficients.
a = [[-.9, .23, 1.1, -.55],
     [.2, -.18, -.4, .35]]

delta = [.55, .45]
```

Output Format Requirements:

1. Emit `<final_answer>` immediately, followed by `<reasoning>`. Put exactly one finite decimal number, with no units or prose, inside `<final_answer>...</final_answer>`.
2. Put the scientific reasoning and requested numerical checkpoints inside `<reasoning>...</reasoning>`. Explain how the mixed derivative accounts for hidden-state marginalization.
3. Do not round intermediate calculations. Give the final result to at least six decimal places.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

01_construct_event_steps.py

Goal
----
Convert supplied turning points into irregular event-indexed observations

```python
def construct_event_steps(points: "np.ndarray", times: "np.ndarray", initial_heading: "np.ndarray", centre: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Construct the known straight steps and their arrival-event covariates.
 
    Parameters
    ----------
    points : np.ndarray
        Finite (n+1,2) turning-point coordinates, with n >= 2 and positive segment lengths.
    times : np.ndarray
        Strictly increasing (n+1,) elapsed hours, in the same coordinate system as clock hour.
    initial_heading : np.ndarray
        Nonzero (2,) vector for the incoming direction at the first segment.
    centre : np.ndarray
        Finite (2,) reference location.
 
    Returns
    -------
    durations, speeds, angles, distances, hours : tuple[np.ndarray, ...]
        Five (n,) arrays. Angle is signed from incoming to current segment in [-pi,pi),
        distances are measured from each segment's arrival point to centre, and
        clock hours are arrival times modulo 24. Durations remain in hours.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 2

02_carved_turn_log_density.py

Goal
----
Evaluate normalized carved von Mises density and its concentration score.

```python
def carved_turn_log_density(angles: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the log angle density and its derivative with respect to kappa.
 
    The source density is exp(kappa*cos(phi))*(1-exp(-width*(1-cos(phi))))
    on [-pi,pi), normalized over that interval. The exact normalization is
    2*pi*[I0(kappa)-exp(-width)*I0(kappa+width)].
 
    Parameters
    ----------
    angles : np.ndarray
        Finite (n,) angles in [-pi,pi). Values at exactly zero yield log density -inf.
    kappa, width : np.ndarray
        Positive finite (2,) state parameters, with width >= 1e-5 and kappa <= 50.
 
    Returns
    -------
    log_density, concentration_score : tuple[np.ndarray, np.ndarray]
        Both (n,2). Score differentiates log density in each state's kappa.
        Use sufficient stability for near-zero angles and moderate widths.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 3

03_gamma_step_log_density.py

Goal
----
Evaluate independent gamma time and speed emissions in event coordinates.

```python
def gamma_step_log_density(durations: "np.ndarray", speeds: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray") -> "np.ndarray":
    """Return state-dependent log gamma densities for observed durations and speeds.
 
    Shape-rate gamma density at x>0 is rate**shape*x**(shape-1)*exp(-rate*x)/Gamma(shape).
    The observed variable is speed, not length; no length-to-speed Jacobian is required.
 
    Parameters
    ----------
    durations, speeds : np.ndarray
        Positive finite (n,) values in hours and coordinate units per hour.
    time_shape, time_rate, speed_shape, speed_rate : np.ndarray
        Positive finite (2,) state-specific shape and rate parameters.
 
    Returns
    -------
    log_density : np.ndarray
        (n,2) sum of independent time and speed log densities.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 4

04_event_transition_matrices.py

Goal
----
Evaluate arrival-event multinomial-logit transitions and a cyclic coefficient derivative.

```python
def event_transition_matrices(distances: "np.ndarray", hours: "np.ndarray", coefficients: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Compute event-indexed transitions and derivative for coefficient beta[0,2].
 
    The two off-diagonal logits are beta[i,0]+beta[i,1]*distance+
    beta[i,2]*cos(pi*hour/12)+beta[i,3]*sin(pi*hour/12).
    The diagonal logit in each row is zero. Transition at index t takes
    state t-1 to state t and uses the arrival event's covariates; index 0
    is returned but is not applied before the first emission.
 
    Parameters
    ----------
    distances, hours : np.ndarray
        Finite (n,) arrival-event covariates, with nonnegative distance.
    coefficients : np.ndarray
        Finite (2,4) logits, rows from origin states 1 and 2.
 
    Returns
    -------
    transitions, derivative : tuple[np.ndarray, np.ndarray]
        Each (n,2,2), row stochastic for transitions. Derivative is with
        respect to the state-1-to-state-2 cosine coefficient only.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 5

05_joint_event_emissions.py

Goal
----
Combine carved angle and gamma time-speed log emissions.

```python
def joint_event_emissions(angle_log: "np.ndarray", gamma_log: "np.ndarray") -> "np.ndarray":
    """Form the conditional independent joint event emission in log coordinates.
 
    Parameters
    ----------
    angle_log, gamma_log : np.ndarray
        Matching (n,2) state-specific log densities, n >= 2.
 
    Returns
    -------
    joint_log : np.ndarray
        (n,2) elementwise log density of angle, duration and speed.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 6

06_scaled_event_forward.py

Goal
----
Compute the stable event HMM log likelihood and filtering probabilities.

```python
def scaled_event_forward(log_emissions: "np.ndarray", transitions: "np.ndarray", initial: "np.ndarray") -> "tuple[float, np.ndarray]":
    """Evaluate the event HMM forward likelihood at fixed parameters.
 
    Initial is fixed independent of model parameters; transition[t]
    connects events t-1 and t only for t >= 1. A row with at least one
    finite log emission is required. Return a finite log likelihood and
    normalized (n,2) filtering probabilities.
 
    Parameters
    ----------
    log_emissions : np.ndarray
        (n,2) finite or -inf log densities, n >= 2, each row having a finite entry.
    transitions : np.ndarray
        (n,2,2) nonnegative row-stochastic matrices.
    initial : np.ndarray
        Positive (2,) fixed probabilities summing to one.
 
    Returns
    -------
    log_likelihood, filters : tuple[float, np.ndarray]
        Log density and (n,2) state-filter distributions.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 7

07_mixed_event_curvature.py

Goal
----
Differentiate the scaled hidden state forward recursion jointly in two parameters.

```python
def mixed_event_curvature(log_emissions: "np.ndarray", concentration_score: "np.ndarray", transitions: "np.ndarray", transition_derivative: "np.ndarray", initial: "np.ndarray") -> float:
    """Return mixed log-likelihood derivative in kappa[0] and beta[0,2].
 
    Kappa[0] changes only the state-1 carved turning emission; beta[0,2]
    changes only the state-1-to-state-2 cyclic cosine transition logit.
    State labels, initial distribution and all other parameters are fixed.
    The derivative is of the marginal log likelihood, including uncertainty
    in hidden states. A result within 1e-6 absolute accuracy is required.
 
    Parameters
    ----------
    log_emissions, concentration_score : np.ndarray
        Matching (n,2) log emission and concentration-score arrays, n >= 2.
    transitions, transition_derivative : np.ndarray
        Matching (n,2,2) transition and derivative arrays.
    initial : np.ndarray
        Positive fixed (2,) probability vector summing to one.
 
    Returns
    -------
    mixed_derivative : float
        d^2 log L / d kappa[0] d beta[0,2].
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```

### Step 8

08_evaluate_event_curvature.py

Goal
----
Orchestrate all model components for the benchmark mixed curvature.

```python
def evaluate_event_curvature(points: "np.ndarray", times: "np.ndarray", heading: "np.ndarray", centre: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray", coefficients: "np.ndarray", initial: "np.ndarray") -> float:
    """Return the fixed-parameter mixed curvature of the event-indexed HMM.
 
    Parameters
    ----------
    points, times, heading, centre : np.ndarray
        Geometric event inputs as specified in construct_event_steps.
    kappa, width : np.ndarray
        Carved turn parameters as specified in carved_turn_log_density.
    time_shape, time_rate, speed_shape, speed_rate : np.ndarray
        Gamma emission shape-rate parameters for durations and speeds.
    coefficients : np.ndarray
        (2,4) off-diagonal transition coefficients.
    initial : np.ndarray
        Fixed (2,) state distribution, independent of coefficients.
 
    Returns
    -------
    mixed_derivative : float
        d^2 log L / d kappa[0] d coefficients[0,2], within 1e-6.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result
```
