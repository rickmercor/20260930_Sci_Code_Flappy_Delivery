# Biology-Ecology-7

## Background

Many ecological interventions act as repeated, nearly instantaneous shocks rather than as continuous forcing, so the community evolves continuously between interventions and is rescaled at scheduled times. Such systems are naturally represented by impulsive differential equations with multiplicative jumps, and persistence cannot in general be inferred from the corresponding untreated equilibrium. Permanence theory studies whether all populations eventually remain uniformly separated from extinction, while boundary invariant sets describe the configurations against which an absent population must successfully invade. For periodically pulsed systems, a useful analysis separates the continuous ecological growth from the discrete intervention and evaluates long-term growth on the invariant pieces of the pulsed boundary dynamics. When a boundary component contains several invariant states linked by the dynamics, the relevant certificate uses one common set of positive weights rather than choosing a different invader for each state. This framework is especially useful when competition, predation, and intervention select for different populations at different frequencies or intensities, because small changes in pulse timing or strength can change which boundary component controls the persistence certificate.

## Problem

A sufficient condition for permanence is that there exists one common positive weight vector whose weighted missing-species growth rate is positive on every invariant boundary Morse component. Failure of this sufficient certificate does not by itself prove extinction or non-persistence.

For the deterministic benchmark margin used below, define a signed, normalized certificate explicitly as follows. For each Morse component M, let E(M) be the set of species that are absent from at least one orbit in M. Species that are resident on every orbit throughout M are excluded from E(M). Let w be a vector indexed by E(M) satisfying

    w_i >= 0,
    sum_{i in E(M)} w_i = 1.

Thus the benchmark uses nonnegative normalized weights; zero weight on an eligible species is allowed. For an orbit O in M, let r_i(O) denote the physical-time long-term growth rate of species i when introduced as a rare invader of O. The component certificate is

    mu(M) =
        max_w  min_{O in M}
            sum_{i in E(M)} w_i r_i(O),

where the maximum is over the normalized nonnegative weights above.

The signed permanence margin is then

    m = min_M mu(M),

where the minimum ranges over all invariant boundary Morse components, including the origin component. Consequently, m may be positive, zero, or negative. A positive value means that this benchmark certificate establishes the sufficient permanence condition; a negative value means only that this particular sufficient certificate is not positive. The benchmark below uses a four-population berry IPM module and replaces a direct search over spray intervals by an inverse control problem: the spray interval is fixed, while a single scale parameter changes the logarithmic strength of every multiplicative pulse. Your task is to compute the unique scale at which the sufficient permanence certificate changes sign. The primary inputs are the ecological coefficients, the reference pulse effects, the fixed interval, and a bracketing interval for the scale; the primary output is one deterministic scalar.

The between-spray dynamics are

\[
\frac{dN_i}{dt}=N_i f_i(z),
\]

for three pest strains \(A,B,C\) and a shared parasitoid \(P\), with

\[
f_i
=
a_i-\sum_{j\in\{A,B,C\}}b_{ij}N_j
-\frac{e_iN_P}{D},
\qquad i\in\{A,B,C\},
\]

\[
f_P
=
-d+\frac{c_AN_A+c_BN_B+c_CN_C}{D},
\]

and

\[
D
=
1+\omega_AN_A+\omega_BN_B+\omega_CN_C.
\]

At each spray,

\[
N_i(k\tau^+)=(1+h_i)N_i(k\tau^-).
\]

The reference pulse vector is

\[
h^0=(-0.4,-0.3,-0.2,-0.2),
\]

in the order \((A,B,C,P)\).

For a trial scale \(q>0\), define the actual pulse vector componentwise by

\[
1+h_i(q)=(1+h_i^0)^q.
\]

Do not replace this transformation by linear scaling of \(h_i\).

Use

\[
\tau=2.0
\]

weeks, with

\[
q\in[1.15,1.20]
\]

for the final threshold search.

The fixed ecological parameters are

```text
a    = (1.3, 1.2, 1.0)

b    = [[1.0, 1.4, 0.4],
        [0.4, 1.0, 1.6],
        [2.0, 0.1, 1.0]]

e    = (1.2, 1.2, 1.0)

c    = (1.5, 1.5, 3.0)

omega= (0.4, 0.15, 0.2)

d    = 0.4

h0   = (-0.4, -0.3, -0.2, -0.2)

tau  = 2.0

[q_lo,q_hi] = [1.15,1.20]
```

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning>. Include the following essential checkpoints needed to justify the deterministic threshold calculation:

1. State the post-spray state convention and the pulse-adjusted physical-time invasion-rate formula.

2. At q=1, identify the complete non-trivial boundary-orbit census and the finest Morse decomposition. Report the post-spray state of each non-trivial boundary periodic orbit.

3. At q=1, report the weighted permanence certificate for each Morse component needed to establish the global minimum, including the common normalized weight vector for every multi-orbit Morse component that contributes to the margin. Identify the Morse component attaining the global margin.

4. Report the signed permanence-margin values at q=1.15 and q=1.20, including their signs, and explain how they establish the supplied root bracket.

5. Report the invasion-rate mappings needed to support the Morse-component certificates, explicitly identifying the boundary orbit and invading species for every reported rate. Report the relevant within-face Floquet radii and the ABC period averages used in the boundary analysis.

6. State the nonlinear pulse-strength transformation h_i(q)=(1+h_i^0)^q-1 and explain that the boundary orbit census, Morse decomposition, invasion rates, weighted certificates, and final permanence margin are recomputed at each trial q.

7. At the critical scale q*, report the limiting Morse component, its common normalized weights, and the invasion-rate mappings that determine its zero weighted certificate.

8. Provide at least one independent numerical consistency check supporting the periodic-orbit or invasion-rate calculation, including the numerical residual or tolerance used for that check. The reported residual must be at or below 1e-9.

Report the numerical intermediate quantities required by these checkpoints, but do not reproduce a full implementation trace. Derive all numerical values from the supplied model and source-supported method; do not assume that any golden numerical values are supplied by the prompt.

A single final numeric answer wrapped in <final_answer>...</final_answer>.

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

single_pest_orbit

Goal
----
Determine the post-spray density of the positive periodic orbit for a single pest strain evolving on a one-dimensional boundary face. The population follows logistic growth between sprays and is instantaneously multiplied by the prescribed pulse factor at each intervention. Identify the positive fixed point of the resulting pulse-to-pulse dynamics rather than estimating it from a finite-time simulation. The calculation must reject parameter regimes in which a positive periodic orbit does not exist.

```python
def single_pest_orbit(a: float, b: float, h: float, tau: float) -> float:
    """Post-spray density of the positive tau-periodic orbit of a pulsed logistic pest strain.

    Parameters
    ----------
    a : float
        Intrinsic growth rate (per week), a > 0.
    b : float
        Intraspecific competition coefficient (per week per unit density), b > 0.
    h : float
        Multiplicative spray factor, h > -1 (N -> (1 + h) N at every spray).
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        N(0+): the density immediately after a spray on the unique positive tau-periodic orbit.

    Raises
    ------
    ValueError
        If a <= 0, b <= 0, tau <= 0, h <= -1, or a*tau + ln(1+h) <= 0 (no positive orbit).
    """

    return 0.0
```

### Step 2

stroboscopic_map

Goal
----
Evaluate one complete post-spray-to-post-spray cycle of the four-species periodically pulsed community. Starting from a specified nonnegative post-spray state, evolve the continuous ecological system for exactly one spray interval and then apply the componentwise multiplicative pulse. Depending on the requested component, return either the corresponding post-spray state coordinate or the continuous per-capita growth integral accumulated during the interval. Preserve the distinction between continuous evolution and the instantaneous intervention.

```python
def stroboscopic_map(params: dict, z0: object, tau: float, which: int) -> float:
    """One-interval (stroboscopic) map of the sprayed four-species community, taken post-spray to post-spray.

    Parameters
    ----------
    params : dict
        Model parameters: 'a' (3 growth rates), 'B' (3x3 competition matrix, row i = effect on strain i),
        'e' (3 attack rates), 'c' (3 parasitoid gains), 'om' (3 handling coefficients), 'd' (parasitoid
        mortality), 'h' (4 spray factors for A, B, C, P; each > -1).
    z0 : array_like, shape (4,)
        Post-spray state (N_A, N_B, N_C, N_P) at t = 0+; all entries >= 0.
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
        0..3 -> the requested component of the post-spray state after one interval;
        4..7 -> the integral of the per-capita rate f_{which-4} over the interval [0, tau].

    Returns
    -------
    float
        The requested component (state after the spray at t = tau, or the rate integral).

    Raises
    ------
    ValueError
        On invalid parameters, negative or non-finite state, tau <= 0 or which outside 0..7.
    """

    return 0.0
```

### Step 3

pest_face_mean_density

Goal
----
Compute the period-averaged density of a specified pest strain on a positive periodic orbit supported entirely on a pest-only boundary face. The calculation must use the pulse-adjusted periodic balance of the selected face rather than substituting an equilibrium density from the unsprayed model. Return the average density over one complete inter-spray interval for the requested resident strain, while validating that the selected face actually supports a positive periodic state.

```python
def pest_face_mean_density(params: dict, present: object, tau: float, which: int) -> float:
    """Period-averaged density of a pest strain on the tau-periodic orbit of a pest-only face.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h' as in the other steps).
    present : sequence of int
        Indices (subset of {0, 1, 2}) of the strains present on the face; non-empty, no repeats.
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
        Index of the strain whose period-averaged density (1/tau) int_0^tau N_which dt is returned.

    Returns
    -------
    float
        The period-averaged density of strain `which` on the orbit of that face.

    Raises
    ------
    ValueError
        On invalid inputs, if `which` is not in `present`, if the parasitoid index 3 is in `present`,
        or if the face carries no positive tau-periodic orbit (the averaged densities are not all positive).
    """

    return 0.0
```

### Step 4

face_orbit_post_pulse

Goal
----
Locate the positive periodic orbit supported on a specified boundary face by solving the fixed-point equations of the post-spray one-interval map. Use the supplied post-spray state as the starting point for the nonlinear solve and return the requested species coordinate at the post-spray section. The result must correspond to a genuinely positive periodic orbit of the pulsed system and satisfy the fixed-point residual requirement in maximum norm.

```python
def face_orbit_post_pulse(params: dict, present: object, guess: object, tau: float, which: int) -> float:
    """Post-spray density of one species on the tau-periodic orbit of a boundary face, by Newton shooting.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    present : sequence of int
        Indices (subset of {0, 1, 2, 3}) of the species present on the face; non-empty, no repeats.
    guess : array_like, shape (len(present),)
        Positive starting densities for the present species (post-spray section).
    tau : float
        Spray interval (weeks), tau > 0.
    which : int
                Species index, one of the values listed in `present`, whose post-spray density on the located orbit is returned.

    Returns
    -------
    float
        Post-spray density N_which(0+) on the tau-periodic orbit of that face.

    Raises
    ------
    ValueError
        On invalid inputs, if `which` is not in `present`, or if the iteration does not converge to a
        positive fixed point of the one-interval map with residual <= 1e-12 (maximum norm).
    """

    return 0.0
```

### Step 5

stroboscopic_floquet_radius

Goal
----
Determine the largest modulus of the Floquet multipliers associated with perturbations that remain within a specified boundary face. Differentiate the post-spray one-interval map at the supplied periodic state, restrict the resulting Jacobian to the coordinates corresponding to species present on the face, and compute its spectral radius. Do not include transverse directions corresponding to absent species.

```python
def stroboscopic_floquet_radius(params: dict, present: object, z_star: object, tau: float) -> float:
    """Spectral radius of the within-face Jacobian of the one-interval map at a face orbit.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    present : sequence of int
        Indices of the species present on the face (subset of {0, 1, 2, 3}).
    z_star : array_like, shape (4,)
        Post-spray state of the tau-periodic orbit (zero for absent species); it must satisfy the
        fixed-point condition of the one-interval map to 1e-8 in the maximum norm.
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        Largest modulus of the eigenvalues of the Jacobian of the one-interval map restricted to the face.

    Raises
    ------
    ValueError
        On invalid inputs, or if z_star is not a fixed point of the one-interval map on that face.
    """

    return 0.0
```

### Step 6

pulsed_invasion_rate

Goal
----
Compute the long-term per-unit-time growth rate of a specified species along a boundary periodic orbit. Integrate that species' continuous per-capita growth rate over one complete inter-spray interval and incorporate the species-specific logarithmic contribution from its own multiplicative pulse. Normalize the complete period growth by the physical spray interval. The calculation must be valid both for resident species and for species absent from the boundary orbit.

```python
def pulsed_invasion_rate(params: dict, z0: object, i: int, tau: float) -> float:
    """Pulse-adjusted long-term per-unit-time growth rate of species i along the tau-periodic orbit through z0.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    z0 : array_like, shape (4,)
        Post-spray state of a tau-periodic boundary orbit (zero for absent species).
    i : int
        Species index in 0..3 (0 = A, 1 = B, 2 = C, 3 = P).
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        r_i = (1/tau) [ int_0^tau f_i(z(t)) dt + ln(1 + h_i) ] along the orbit through z0.

    Raises
    ------
    ValueError
        On invalid inputs, or if z0 is not a fixed point of the one-interval map to 1e-8.
    """

    return 0.0
```

### Step 7

boundary_census

Goal
----
Enumerate the positive periodic population states supported on the proper non-empty boundary faces of the four-species pulsed community. Determine which faces actually carry positive periodic orbits under the current pulse vector, rather than assuming that every orbit of the corresponding untreated system persists after intervention. Count the non-trivial boundary periodic orbits while excluding the origin, and maintain consistency across the face-level orbit calculations.

```python
def boundary_census(params: dict, tau: float) -> float:
    """Number of non-trivial tau-periodic orbits on the boundary of the sprayed four-species community.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        The number of boundary faces (proper non-empty subsets of the four species) carrying a positive
        tau-periodic orbit; the origin is not counted.

    Raises
    ------
    ValueError
        On invalid inputs or on any internal inconsistency of the orbit search.
    """

    return 0.0
```

### Step 8

permanence_margin

Goal
----
Construct the finest Morse decomposition of the pulsed boundary attractor and determine the permanence certificate associated with each Morse component. For each component containing multiple boundary invariant orbits, identify a single common probability weighting over the relevant species growth rates and solve the resulting max-min optimization problem. The component value is the largest guaranteed weighted growth shared across all invariant orbits represented by that component. Return the minimum component value over the complete boundary decomposition.

```python
def permanence_margin(params: dict, tau: float) -> float:
    """Permanence margin m(tau) of the sprayed four-species community.

    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        m(tau) = min over the pieces of the finest Morse decomposition of the boundary attractor of
        mu_k(tau), where mu_k is the largest value of min over the orbits of the piece of the weighted
        pulse-adjusted growth rates, the weights ranging over probability vectors supported on the species
        absent from at least one orbit of the piece (the origin piece uses the largest single-species rate).

    Raises
    ------
    ValueError
        On invalid inputs or on any internal inconsistency of the census, the decomposition or the
        linear programmes.
    """

    return 0.0
```

### Step 9

critical_pulse_scale

Goal
----
Determine the unique pulse-strength scale at which the signed normalized permanence margin changes sign. The spray interval and ecological parameters remain fixed, while a single dimensionless scale modifies the logarithmic strength of every multiplicative pulse relative to the reference pulse vector. For every trial scale, independently rebuild the pulsed boundary dynamics, identify the existing periodic boundary orbits, reconstruct the finest Morse decomposition, recompute all pulse-adjusted invasion rates, solve the associated weighted max-min certificates, and evaluate the resulting permanence margin. The Morse decomposition is compared with the reference decomposition at each trial value; a change in decomposition invalidates the supplied fixed-bracket calculation. Locate the zero of the fully recomputed margin within the supplied bracket to the required absolute tolerance.

```python
def critical_pulse_scale(params: dict, tau: float, q_lo: float, q_hi: float) -> float:
    """Locate the unique q where the fully recomputed permanence margin crosses zero.

    Parameters
    ----------
    params : dict
        Base model parameters. The entries in ``h`` are the reference pulse effects h_i^0.
    tau : float
        Fixed spray interval (weeks), tau > 0.
    q_lo, q_hi : float
        Positive bracket with opposite signs of the recomputed permanence margin.

    Returns
    -------
    float
        Critical pulse-strength scale q*, located to absolute tolerance 1e-12.

    Raises
    ------
    ValueError
        If inputs are invalid, the bracket has no sign change, the boundary Morse
        decomposition changes inside the bracket, or the assembled chain is internally
        inconsistent.
    """
    return 0.0
```
