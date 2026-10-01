# Biology-Ecology-27

## Background

A recent line of work gives a sufficient condition for permanence (uniform persistence of every species) in $n$-species Kolmogorov communities whose densities are multiplied by fixed factors at regularly spaced instants. The condition is a weighted long-term growth-rate condition on the pieces of a Morse decomposition of the boundary attractor of the pulsed system, with the between-pulse dynamics and the pulse entering as separate contributions. Reproduce that criterion on the bespoke integrated-pest-management configuration below and return the critical spray interval at which the certification appears.
 
From the source you must determine: (i) the discrete-time map through which the boundary invariant sets of a periodically pulsed community are defined and located, including where in the pulse cycle its section is taken; (ii) the exact form of the long-term growth rate of a species along such a set — how the between-pulse dynamics and the pulse are combined and normalised, and whether the pulse acting on a species that is absent from the set enters that species' rate; (iii) how the rates of several species are combined on a piece that carries more than one invariant set, and which weight vectors are admissible; (iv) the identity that fixes the period average of a species present on a periodic boundary orbit; (v) the conditions under which each candidate boundary set exists in the pulsed community.
 
**Frozen configuration.** Three strains A, B, C of a berry pest compete on the same fruit and are attacked by a shared parasitoid wasp P with a saturating multi-host response; a broad-spectrum pesticide is sprayed every $\tau$ weeks and kills a fixed fraction of each population. With $z=(N_A,N_B,N_C,N_P)$ (dimensionless densities; species indices $0,1,2,3$ in that order), between sprays $dN_i/dt=N_i f_i(z)$ with
 
$$f_i = a_i - \sum_{j\in\{A,B,C\}} b_{ij}N_j - \frac{e_i N_P}{D}\ (i=A,B,C),\qquad f_P = -d + \frac{c_A N_A + c_B N_B + c_C N_C}{D},\qquad D = 1+\omega_A N_A+\omega_B N_B+\omega_C N_C,$$
 
and at every spray instant $t=k\tau$ ($k=1,2,\dots$) each density jumps instantaneously, $N_i(k\tau^+)=(1+h_i)\,N_i(k\tau^-)$. Parameters:
 
```
a    = (1.3, 1.2, 1.0)        # week^-1, intrinsic growth of strains A, B, C
b    = [[1.0, 1.4, 0.4],      # week^-1 per unit density; row i = effect of (A, B, C) on strain i
        [0.4, 1.0, 1.6],
        [2.0, 0.1, 1.0]]
e    = (1.2, 1.2, 1.0)        # week^-1 per unit parasitoid density, attack on A, B, C
c    = (1.5, 1.5, 3.0)        # week^-1 per unit host density, parasitoid gain from A, B, C
omega = (0.4, 0.15, 0.2)      # per unit density, handling saturation on A, B, C
d    = 0.4                    # week^-1, parasitoid mortality
h    = (-0.4, -0.3, -0.2, -0.2)   # spray removes 40 % of A, 30 % of B, 20 % of C, 20 % of P
tau_0            = 2.0        # weeks, reference spray interval
[tau_lo, tau_hi] = [1.2, 3.0] # weeks, bracket for the threshold
```
 
**Reported quantity.** Let $\{M_k\}$ be the finest Morse decomposition of the boundary attractor of the pulsed community in the sense of the source at spray interval $\tau$ (each piece a chain-transitive isolated invariant set of the pulsed boundary dynamics), let $r_i(\mu;\tau)$ be the source's long-term per-unit-time growth rate of species $i$ along an ergodic invariant measure $\mu$ carried by $M_k$, and let $S_k$ be the set of species absent from at least one ergodic measure of $M_k$. Define
 
$$\mu_k(\tau)=\max_{p\in\Delta(S_k)}\ \min_{\mu\in\mathrm{Erg}(M_k)}\ \sum_i p_i\, r_i(\mu;\tau),$$
 
the maximum over probability vectors supported on $S_k$ (for the piece consisting of the origin, $S_k$ is the full species set and $\mu_k$ is the largest single-species rate), and the permanence margin $m(\tau)=\min_k \mu_k(\tau)$; the criterion certifies permanence when $m(\tau)>0$. On the bracket, $m(\tau_{lo})<0<m(\tau_{hi})$ and $m$ has exactly one zero $\tau^\ast$. That zero — the smallest spray interval for which the sprayed four-species community is certified permanent — is the quantity to return.
 
**Benchmark convention (not a paper parameter).** float64 throughout; no randomness. Rates are per unit time of the sprayed system, i.e. per-interval totals divided by $\tau$ (not by $\tau+1$, and not left as per-interval totals). All densities on periodic orbits are reported at the post-spray instant $k\tau^+$. Between-spray trajectories and the time integrals of per-capita rates along them must be converged to an absolute accuracy of $10^{-10}$ (for reference, DOP853 with rtol $=10^{-13}$, atol $=10^{-16}$, and classical fixed-step RK4 with at least 2000 steps per interval both reach $10^{-14}$ on this configuration). Every periodic orbit — whether or not it attracts within its face — must satisfy its defining fixed-point condition to a residual of at most $10^{-12}$ in the maximum norm (any convergent iteration; the starting point is at your discretion). The weight vectors are optimised exactly (a linear programme solved to its optimal vertex). The zero $\tau^\ast$ is located by a bracketing root search on $[\tau_{lo},\tau_{hi}]$ to an absolute tolerance of $10^{-12}$. Report $\tau^\ast$ in weeks to at least six significant figures.
 
Do not hard-code the number of boundary orbits, the number of pieces or the identity of the limiting piece; do not decide the existence of a boundary orbit from the unsprayed system; do not replace a piece that carries several orbits by its orbits taken one at a time; do not approximate a period integral by evaluating its integrand at period-averaged densities; do not replace the located zero by interpolation on a coarse grid; do not use tolerances looser than those stated.
 
In your reasoning, report these quantities from your run alongside the final answer, all at $\tau_0=2.0$ unless stated: (1) the number of non-trivial boundary periodic orbits and the number of pieces of the decomposition (origin included); (2) the post-spray density on each single-strain orbit; (3) the post-spray densities $(N_A,N_B,N_C)$ on the three-strain orbit and $(N_i,N_P)$ on each strain–parasitoid orbit; (4) the largest eigenvalue modulus of the within-face Jacobian of the one-interval map at the three-strain orbit and at each strain–parasitoid orbit; (5) the period-averaged densities on the three-strain orbit; (6) the rates $r_P$ on the three single-strain orbits and on the three-strain orbit, and the six rates of the absent strains on the strain–parasitoid orbits; (7) $\mu_k$ and the optimal weights for every piece that carries more than one orbit; (8) the margin $m(\tau_0)$ and the piece at which it is attained; (9) $m(\tau_{lo})$, $m(\tau_{hi})$ and the located zero $\tau^\ast$; (10) at $\tau^\ast$, the optimal weights of the piece at which the margin is attained and the rates of the absent species on each of its orbits, and one exact identity you checked the run against.
 
Return the critical spray interval $\tau^\ast$ (weeks) produced by this deterministic benchmark.
 
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, report all ten requested diagnostic groups concisely, together with the source-derived map, growth-rate, common-weight and orbit-existence conventions needed to justify them. Compact labelled lines are permitted. Do not reproduce the supplied input matrices or coefficient vectors, per-iteration paths, root-search histories, or full candidate tables.
 
## Background
 
Many management interventions act on populations as brief, regularly repeated shocks rather than as continuous pressure: pesticide sprays in orchards and berry fields, cycles of chemotherapy, rounds of mass drug administration. Between interventions the community follows its own continuous-time ecology; at each intervention a fixed fraction of every affected population is removed at once. Impulsive differential equations with multiplicative pulses are the standard description of such regimes, and their outcome cannot be read off the untreated model: the same intervention can favour or eliminate a species depending on how its timing interacts with competition and predation during the intervals between shocks.
 
The configuration studied here is an integrated pest-management module with intransitive competition. Three strains of one pest compete on the same fruit with cyclic dominance — B excludes A, C excludes B and A excludes C when any two meet alone, the rock–paper–scissors structure familiar from toxin-producing, resistant and sensitive strains — and all three are attacked by a shared parasitoid whose attack rate saturates with total host density, so the strains also interact indirectly through the shared enemy (apparent competition). A broad-spectrum pesticide sprayed every $\tau$ weeks removes fixed fractions of every strain and of the parasitoid; the strains differ in their sensitivity to the spray, and the parasitoid is a poorer host on the fast-growing sensitive strain than on the resistant one. The management objective is persistence of all four populations: the parasitoid must persist because it is the biocontrol agent, and no strain may be lost, because the loss of one strain removes the check that keeps the strain it dominates in balance.
 
Permanence — every population bounded away from extinction uniformly in the initial state — is the natural formalisation of that objective. For continuous-time Kolmogorov systems the classical route is the weighted-average-growth-rate criterion: decompose the invariant sets on the boundary (where at least one species is absent) into the pieces of a Morse decomposition and require, for each piece, a common positive weight vector under which the weighted long-term growth rate of the species along every invariant measure carried by the piece is positive. The weights are what distinguishes a heteroclinic cycle from its constituent equilibria: on a cycle every equilibrium can be invaded by some species, yet the cycle as a whole attracts unless the products of the positive rates around it dominate the products of the negative ones — a condition that only a weight vector common to the whole cycle detects.
 
The task instantiates this programme on the sprayed four-species community. Frequent spraying suppresses the parasitoid and tilts the cycle of strain–parasitoid orbits towards attraction; infrequent spraying lets the certification hold. The spray interval at which the certification appears — the smallest interval for which all four populations are certified permanent — is the reported scalar, and it requires: the pulsed periodic orbits located as fixed points of the one-interval map to a tight residual (the three-strain orbit cannot be found by simulation), the long-term growth rates of every absent species along every orbit, the grouping of the orbits into chain-transitive pieces, the exact optimisation of the weight vector on each piece, and a bracketing root search in $\tau$.

## Problem

A recent line of work gives a sufficient condition for permanence (uniform persistence of every species) in $n$-species Kolmogorov communities whose densities are multiplied by fixed factors at regularly spaced instants. The condition is a weighted long-term growth-rate condition on the pieces of a Morse decomposition of the boundary attractor of the pulsed system, with the between-pulse dynamics and the pulse entering as separate contributions. Reproduce that criterion on the bespoke integrated-pest-management configuration below and return the critical spray interval at which the certification appears.
 
From the source you must determine: (i) the discrete-time map through which the boundary invariant sets of a periodically pulsed community are defined and located, including where in the pulse cycle its section is taken; (ii) the exact form of the long-term growth rate of a species along such a set — how the between-pulse dynamics and the pulse are combined and normalised, and whether the pulse acting on a species that is absent from the set enters that species' rate; (iii) how the rates of several species are combined on a piece that carries more than one invariant set, and which weight vectors are admissible; (iv) the identity that fixes the period average of a species present on a periodic boundary orbit; (v) the conditions under which each candidate boundary set exists in the pulsed community.
 
**Frozen configuration.** Three strains A, B, C of a berry pest compete on the same fruit and are attacked by a shared parasitoid wasp P with a saturating multi-host response; a broad-spectrum pesticide is sprayed every $\tau$ weeks and kills a fixed fraction of each population. With $z=(N_A,N_B,N_C,N_P)$ (dimensionless densities; species indices $0,1,2,3$ in that order), between sprays $dN_i/dt=N_i f_i(z)$ with
 
$$f_i = a_i - \sum_{j\in\{A,B,C\}} b_{ij}N_j - \frac{e_i N_P}{D}\ (i=A,B,C),\qquad f_P = -d + \frac{c_A N_A + c_B N_B + c_C N_C}{D},\qquad D = 1+\omega_A N_A+\omega_B N_B+\omega_C N_C,$$
 
and at every spray instant $t=k\tau$ ($k=1,2,\dots$) each density jumps instantaneously, $N_i(k\tau^+)=(1+h_i)\,N_i(k\tau^-)$. Parameters:
 
```
a    = (1.3, 1.2, 1.0)        # week^-1, intrinsic growth of strains A, B, C
b    = [[1.0, 1.4, 0.4],      # week^-1 per unit density; row i = effect of (A, B, C) on strain i
        [0.4, 1.0, 1.6],
        [2.0, 0.1, 1.0]]
e    = (1.2, 1.2, 1.0)        # week^-1 per unit parasitoid density, attack on A, B, C
c    = (1.5, 1.5, 3.0)        # week^-1 per unit host density, parasitoid gain from A, B, C
omega = (0.4, 0.15, 0.2)      # per unit density, handling saturation on A, B, C
d    = 0.4                    # week^-1, parasitoid mortality
h    = (-0.4, -0.3, -0.2, -0.2)   # spray removes 40 % of A, 30 % of B, 20 % of C, 20 % of P
tau_0            = 2.0        # weeks, reference spray interval
[tau_lo, tau_hi] = [1.2, 3.0] # weeks, bracket for the threshold
```
 
**Reported quantity.** Let $\{M_k\}$ be the finest Morse decomposition of the boundary attractor of the pulsed community in the sense of the source at spray interval $\tau$ (each piece a chain-transitive isolated invariant set of the pulsed boundary dynamics), let $r_i(\mu;\tau)$ be the source's long-term per-unit-time growth rate of species $i$ along an ergodic invariant measure $\mu$ carried by $M_k$, and let $S_k$ be the set of species absent from at least one ergodic measure of $M_k$. Define
 
$$\mu_k(\tau)=\max_{p\in\Delta(S_k)}\ \min_{\mu\in\mathrm{Erg}(M_k)}\ \sum_i p_i\, r_i(\mu;\tau),$$
 
the maximum over probability vectors supported on $S_k$ (for the piece consisting of the origin, $S_k$ is the full species set and $\mu_k$ is the largest single-species rate), and the permanence margin $m(\tau)=\min_k \mu_k(\tau)$; the criterion certifies permanence when $m(\tau)>0$. On the bracket, $m(\tau_{lo})<0<m(\tau_{hi})$ and $m$ has exactly one zero $\tau^\ast$. That zero — the smallest spray interval for which the sprayed four-species community is certified permanent — is the quantity to return.
 
**Benchmark convention (not a paper parameter).** float64 throughout; no randomness. Rates are per unit time of the sprayed system, i.e. per-interval totals divided by $\tau$ (not by $\tau+1$, and not left as per-interval totals). All densities on periodic orbits are reported at the post-spray instant $k\tau^+$. Between-spray trajectories and the time integrals of per-capita rates along them must be converged to an absolute accuracy of $10^{-10}$ (for reference, DOP853 with rtol $=10^{-13}$, atol $=10^{-16}$, and classical fixed-step RK4 with at least 2000 steps per interval both reach $10^{-14}$ on this configuration). Every periodic orbit — whether or not it attracts within its face — must satisfy its defining fixed-point condition to a residual of at most $10^{-12}$ in the maximum norm (any convergent iteration; the starting point is at your discretion). The weight vectors are optimised exactly (a linear programme solved to its optimal vertex). The zero $\tau^\ast$ is located by a bracketing root search on $[\tau_{lo},\tau_{hi}]$ to an absolute tolerance of $10^{-12}$. Report $\tau^\ast$ in weeks to at least six significant figures.
 
Do not hard-code the number of boundary orbits, the number of pieces or the identity of the limiting piece; do not decide the existence of a boundary orbit from the unsprayed system; do not replace a piece that carries several orbits by its orbits taken one at a time; do not approximate a period integral by evaluating its integrand at period-averaged densities; do not replace the located zero by interpolation on a coarse grid; do not use tolerances looser than those stated.
 
In your reasoning, report these quantities from your run alongside the final answer, all at $\tau_0=2.0$ unless stated: (1) the number of non-trivial boundary periodic orbits and the number of pieces of the decomposition (origin included); (2) the post-spray density on each single-strain orbit; (3) the post-spray densities $(N_A,N_B,N_C)$ on the three-strain orbit and $(N_i,N_P)$ on each strain–parasitoid orbit; (4) the largest eigenvalue modulus of the within-face Jacobian of the one-interval map at the three-strain orbit and at each strain–parasitoid orbit; (5) the period-averaged densities on the three-strain orbit; (6) the rates $r_P$ on the three single-strain orbits and on the three-strain orbit, and the six rates of the absent strains on the strain–parasitoid orbits; (7) $\mu_k$ and the optimal weights for every piece that carries more than one orbit; (8) the margin $m(\tau_0)$ and the piece at which it is attained; (9) $m(\tau_{lo})$, $m(\tau_{hi})$ and the located zero $\tau^\ast$; (10) at $\tau^\ast$, the optimal weights of the piece at which the margin is attained and the rates of the absent species on each of its orbits, and one exact identity you checked the run against.
 
Return the critical spray interval $\tau^\ast$ (weeks) produced by this deterministic benchmark.
 
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, report all ten requested diagnostic groups concisely, together with the source-derived map, growth-rate, common-weight and orbit-existence conventions needed to justify them. Compact labelled lines are permitted. Do not reproduce the supplied input matrices or coefficient vectors, per-iteration paths, root-search histories, or full candidate tables.

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

01_single_pest_orbit

Goal
----
Post-spray density of the single-strain tau-periodic orbit.

Between sprays a lone pest strain follows dN/dt = N (a - b N); at every spray its density is multiplied by (1 + h), h > -1. This step returns the density N(0+) immediately after a spray on the unique positive tau-periodic orbit of that pulsed logistic population, in closed form. It validates positivity of a, b, tau, the constraint h > -1 and the existence condition a tau + ln(1+h) > 0 (a ValueError is raised otherwise). Deliberately excluded: any interaction with other species and any numerical time stepping.

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

02_stroboscopic_map

Goal
----
One-interval (stroboscopic) map of the sprayed four-species community.

Given a post-spray state z0 = (NA, NB, NC, NP), this step integrates the between-spray Kolmogorov system over one interval of length tau with DOP853 (rtol 1e-13, atol 1e-16), applies the spray z -> (1+h) z, and returns either one component of the resulting post-spray state (which = 0..3) or the time integral over [0, tau] of one per-capita rate f(i) along the trajectory (which = 4..7), the latter obtained from an augmented state so that no separate quadrature is needed. It validates the parameter dictionary, the non-negativity of z0, tau > 0 and the range of which. Deliberately excluded: locating fixed points and the spray term ln(1+h) of the growth rates (added in step 6).

```python
def stroboscopic_map(params: dict, z0: "np.ndarray", tau: float, which: int) -> float:
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

03_pest_face_mean_density

Goal
----
Period-averaged densities on a pest-only face orbit from the averages identity.

On any face that contains pest strains only, a tau-periodic orbit must satisfy, for every present strain i, a(i) tau + ln(1 + h(i)) = sum over present j of b(ij) times the integral of N(j) over one interval. This step solves that linear system for the period averages (1/tau) int N(j) dt of the present strains and returns the average of strain which. A non-positive solution component means that the face carries no positive tau-periodic orbit, and a ValueError is raised; it is also raised when which is not present or the parasitoid is listed. Deliberately excluded: the orbit itself (its post-spray state needs shooting, step 4) and any face containing the parasitoid (its saturating response has no such identity).

```python
def pest_face_mean_density(params: dict, present: tuple[int, ...], tau: float, which: int) -> float:
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

04_face_orbit_post_pulse

Goal
----
Newton shooting for the tau-periodic orbit of a boundary face.

Given the list of present species, a positive starting guess for their post-spray densities and tau, this step solves pi(z) = z on that face by a damped Newton iteration that uses the Jacobian of the one-interval map obtained from the variational equations integrated together with the flow (DOP853, rtol 1e-13, atol 1e-16); the iteration stops when the fixed-point residual is at most 1e-12 in the maximum norm and the post-spray density of species which on the located orbit is returned. Absent species are held at zero, the iterate is kept positive, and a ValueError is raised for invalid inputs or non-convergence. Deliberately excluded: any statement about stability (step 5) or invasibility (step 6).

```python
def face_orbit_post_pulse(
    params: dict,
    present: tuple[int, ...],
    guess: "np.ndarray",
    tau: float,
    which: int,
) -> float:
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

05_stroboscopic_floquet_radius

Goal
----
Largest Floquet multiplier modulus of a face orbit within its face.

For a tau-periodic orbit given by its post-spray state (zero for absent species), this step checks the fixed-point condition of the one-interval map to 1e-8, restricts the Jacobian of that map to the present species and returns the largest modulus of its eigenvalues. A value below one means that the orbit attracts within its face (it is the face attractor when it is the only candidate), a value above one that it does not. Deliberately excluded: the transverse (invasion) directions, which are handled by the growth rates of step 6 and not by eigenvalues.

```python
def stroboscopic_floquet_radius(
    params: dict,
    present: tuple[int, ...],
    z_star: "np.ndarray",
    tau: float,
) -> float:
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

06_pulsed_invasion_rate

Goal
----
Pulse-adjusted long-term growth rate of one species along a boundary orbit.

For the tau-periodic orbit through a post-spray state z0 (checked to be a fixed point of the one-interval map to 1e-8) and a species index i, this step returns r(i) = (1/tau) [ int over [0, tau] of f(i) along the orbit + ln(1 + h(i)) ]: the between-spray integral of the per-capita rate of species i evaluated along the orbit plus the logarithm of its own spray factor, divided by the spray interval. For a species present on the orbit the value is zero to round-off. Deliberately excluded: any combination of the rates across species or orbits (steps 8 and 9).

```python
def pulsed_invasion_rate(params: dict, z0: "np.ndarray", i: int, tau: float) -> float:
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

07_boundary_census

Goal
----
Census of the boundary tau-periodic orbits.

This step enumerates the fifteen proper non-empty subsets of the four species and decides which of them carry a positive tau-periodic orbit: a pest-only face carries one exactly when the averages identity of step 3 has a positive solution (the orbit is then shot as in step 4 and verified); a face with one strain and the parasitoid carries one when the parasitoid's rate on that strain's orbit is positive (the orbit is shot from the unsprayed face equilibrium); a face with two strains and the parasitoid is examined only when each strain can invade the orbit of the face without it, and the shot fixed point is accepted only if it is positive. It returns the number of orbits found (the origin is not counted) and raises a ValueError on inconsistency. Deliberately excluded: the grouping of the orbits into pieces (step 8).

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

08_permanence_margin

Goal
----
Permanence margin of the sprayed community.

This step builds the census of step 7 with the rates of step 6 along every orbit, constructs the invasion graph (from each orbit, every species with positive rate is introduced at density 1e-4 and the one-interval map is iterated until a known boundary orbit is reached), takes its strongly connected components together with the origin as the pieces of the finest Morse decomposition of the boundary attractor, and evaluates for each piece mu = max over probability vectors p supported on the species absent from at least one orbit of the piece of min over the orbits of the piece of sum p(i) r(i): a small linear programme whose optimal vertex is re-solved exactly. The origin piece uses the largest single-species rate. It returns m(tau) = min over pieces of mu. Deliberately excluded: the root search in tau (step 9).

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

09_critical_spray_interval

Goal
----
Critical spray interval of the sprayed four-species community (orchestrator).

This final step assembles the whole pipeline. At the reference interval tau-zero and at both bracket ends it builds the census from the earlier steps (closed-form single-strain orbits checked against the stroboscopic map, averages identity on the pest faces checked against the quadrature, Newton-shot face orbits, within-face multipliers of the strain-parasitoid orbits, pulse-adjusted rates with the resident-zero identity, census cross-check), determines the Morse pieces, verifies that the decomposition is the same at tau-lo, tau-zero and tau-hi, and evaluates the margin through the weighted piece values; the same chain with the orbits re-shot at every tau is used inside a bracketing root search (brentq, absolute tolerance 1e-12) on [tau-lo, tau-hi], where m(tau-lo) and m(tau-hi) must have opposite signs, and the located zero tau-star is returned. The margin step is used only as an independent cross-check at tau-zero and at tau-star; a ValueError is raised on any inconsistency. Deliberately excluded: anything beyond the sufficient criterion (a negative margin is not a proof of extinction).

```python
def critical_spray_interval(params: dict, tau0: float, tau_lo: float, tau_hi: float) -> float:
    """Critical spray interval tau* at which the permanence margin of the sprayed community vanishes.
 
    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau0 : float
        Reference spray interval (weeks) at which the full boundary analysis is assembled and cross-checked.
    tau_lo, tau_hi : float
        Bracket (weeks) with m(tau_lo) and m(tau_hi) of opposite signs, 0 < tau_lo < tau_hi.
 
    Returns
    -------
    float
        The zero tau* of the permanence margin m(tau) in [tau_lo, tau_hi], located to 1e-12.
 
    Raises
    ------
    ValueError
        On invalid inputs, a bracket that does not enclose a sign change, a Morse decomposition that is not
        the same at tau_lo, tau0 and tau_hi, or any inconsistency between the assembled chain and the
        independent margin evaluation.
    """
    return 0.0
```
