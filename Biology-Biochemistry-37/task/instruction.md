# Biology-Biochemistry-37

## Background

Cellular signal transduction proceeds through cascades of enzymatic reactions in which each component is switched between two states, typically by a kinase and an opposing phosphatase, and the switched component in turn acts on the next component in the chain. Such feed-forward pathways, from mitogen-activated protein kinase cascades to multi-stage phosphorelays, transmit information as a sequence of discrete node-to-node activations. Because each step is an enzyme-catalysed conversion, its kinetics saturate: the rate at which a node is converted depends hyperbolically on how much of it remains in the unconverted form, in the manner of Michaelis-Menten kinetics, and the drive comes from the upstream node's state.

When the activity of every node is written as a signed variable that is +1 when the node is fully in one form and -1 when it is fully in the other, a uniform chain reduces at rest to a single autonomous equation whose equilibria are the two saturated states and, over a range of activation bias, an interior unstable state that separates their basins of attraction. In that bistable regime a sustained stimulus applied at the top of the chain does not merely relax the pathway; it launches a front that switches nodes one after another and advances downstream at a well-defined speed, much as a pushed front propagates through a bistable reaction-diffusion medium. Outside the bistable window the chain is locked in one state and no sustained wave exists. The speed of the front, and hence the delay between stimulus and response many steps downstream, is set by the interplay between the bias toward the target state, the saturation of the kinetics and the timescale of each conversion.

Real pathways are not uniform. Enzyme abundances, catalytic efficiencies and binding affinities differ from step to step, and these differences act as local accelerants or bottlenecks that distort the front and make its speed, measured in nodes per unit time, fluctuate strongly along the chain. Predictable signal timing then requires a way of separating the intrinsic wave dynamics from the parameter-driven irregularity. Since the transition region of a front spans only a handful of nodes, the delay that a heterogeneous chain imposes can be attributed edge by edge, which suggests describing the pathway in a coordinate whose local spacing reflects how quickly each edge would let a front pass. Assessing whether such a description recovers a uniformly moving wave, and where it breaks down as heterogeneity becomes extreme, requires quantitative measures of how the front's speed and shape vary in time.

## Problem

Intracellular signalling cascades relay a stimulus through a chain of enzymatic activation steps, each converting a downstream component between two forms such as phosphorylated and dephosphorylated states. Signal fidelity depends on the local interaction kinetics of every step, yet real cascades are kinetically heterogeneous: timescales and saturation properties vary from edge to edge, so a front of activation stutters as it advances and the delay between stimulus and downstream response becomes unpredictable. A recent modelling framework treats such a feed-forward pathway as a discrete chain of nonlinear, Michaelis-Menten-type node equations, characterises the travelling activation waves it supports, and introduces a spatial rescaling of the node coordinate that absorbs edge-to-edge kinetic variation so that the wave moves at a nearly constant speed in the rescaled coordinate. The inputs of the method are the per-edge kinetic parameters, the boundary stimulus and the resting state; its output is a description of the front's motion in the original and in the rescaled coordinate, summarised by how strongly the front's speed fluctuates.

Consider a pathway of N = 120 nodes with activities x_i in [-1, 1] (i = 1, ..., N), where +1 and -1 denote saturation in the active and the inactive form. Node i is driven by node i - 1 through the edge equation

dx_i/dt = (1 + phi_i)/4 (1 + x_{i-1}) alpha_i beta_i (1 - x_i) / (2 beta_i - (1 + x_i)) - (1 - phi_i)/4 (1 - x_{i-1}) alpha_i beta_i (1 + x_i) / (2 beta_i - (1 - x_i)),

with alpha_i the timescale of edge i, beta_i its saturation parameter, expressed through B_i = 2 beta_i - 1 > 1, and phi_i in [-1, 1] the activation bias. Node 1 is driven by a constant upstream input x_0 = +1, and every node starts at rest at x_i(0) = -1. Take alpha_i = 1 + 4 (i - 1)/(N - 1), B_i = 3 for i <= 60 and B_i = 8 for i > 60, and phi_i = 0 for every edge; with these values every edge lies in the bistable regime in which a sustained front can propagate. Integrate the node equations accurately (relative tolerance 1e-8 or tighter), sample the pathway every delta t = 1 time unit, and stop sampling at the first sample at which the terminal node has moved by more than 10^-4 from its resting value. The activity profile at a sample is interpolated piecewise-linearly between consecutive nodes, with the upstream input counted as node 0 at position zero in whichever node coordinate is in use; the wave centre at a sample is the position at which that interpolated profile first crosses zero.

Estimate the instantaneous propagation speed of the front on every sampling interval with the framework's shape-matching definition of instantaneous wave velocity, both in the node-index coordinate and in the framework's rescaled coordinate, in which the spacing between consecutive nodes is inversely proportional to the intrinsic propagation speed of the connecting edge and the whole pathway is mapped onto the unit interval. Take each edge's intrinsic speed to be the asymptotic propagation speed, in nodes per unit time, of the front launched by the same input into the same resting state in an infinitely long uniform pathway in which every edge carries that edge's parameters, that is, the speed at which the front's profile translates without change of shape once the boundary transient has decayed; obtain it to a relative accuracy of 1e-9 or better, and note that the intrinsic speed of an edge with timescale alpha_i is alpha_i times that of the unit-timescale edge with the same B_i and phi_i, since alpha only rescales time. Express both speed series as fractions of the respective total pathway length per unit time, and evaluate the velocity integral square error of each series over the post-transient window, defined as the sampling intervals whose starting sample has its wave centre, as a fraction of the rescaled pathway length, in [0.15, 0.85): the trapezoidal integral over the window of the squared deviation of the speed from its window mean. In your reasoning, report the number of sampling intervals in the run, the two unit-timescale intrinsic speeds, the rescaled position of node 60, the first and last sampling intervals of the window, the two window-mean speeds and the two integral square errors, and state in one sentence each what the framework finds about how the asymptotic front speed of a uniform pathway depends on the saturation parameter B and on the activation bias within the bistable window, and about how its rescaling affects velocity fluctuations as the degree of kinetic heterogeneity grows.

Report the ratio of the velocity integral square error in the rescaled coordinate to that in the node-index coordinate, as a single number.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

uniform_steady_states

Goal
----
Return the interior uniform equilibrium and the bistability threshold of the cascade model for a single edge with bias phi and saturation parameter B, as the array [xi, phi_c]. Both follow in closed form from the uniform reduction of the node equation, in which every node sits at the same activity level.

```python
def uniform_steady_states(phi: float, B: float) -> "np.ndarray":
    """Return the interior uniform equilibrium and the bistability threshold of the cascade model for a single edge with bias phi and saturation parameter B, as the array [xi, phi_c]. Both follow in closed form from the uniform reduction of the node equation, in which every node sits at the same activity level.

    Parameters
    ----------
    phi : float
        Bias parameter of the edge, in [-1, 1].
    B : float
        Saturation parameter B = 2 beta - 1 of the edge, greater than 1.

    Returns
    -------
    out : np.ndarray
        Array [xi, phi_c]: interior equilibrium and bistability threshold.

    Raises
    ------
    ValueError
        If phi is outside [-1, 1] or B is not greater than 1.
    """
    return out
```

### Step 2

cascade_rate

Goal
----
Evaluate the right-hand side of the node equations of the cascade for the current activity vector x, returning dx_i/dt for every node. Node 1 is driven by the constant upstream input x_in and node i > 1 by node i - 1; the edge feeding node i has parameters alpha_i, B_i = 2 beta_i - 1 and phi_i. Use exactly the activation/inactivation form of the node equation given in the task.

```python
def cascade_rate(x: "np.ndarray", x_in: float, alpha: "np.ndarray", B: "np.ndarray",
                         phi: "np.ndarray") -> "np.ndarray":
    """Evaluate the right-hand side of the node equations of the cascade for the current activity vector x, returning dx_i/dt for every node. Node 1 is driven by the constant upstream input x_in and node i > 1 by node i - 1; the edge feeding node i has parameters alpha_i, B_i = 2 beta_i - 1 and phi_i. Use exactly the activation/inactivation form of the node equation given in the task.

    Parameters
    ----------
    x : np.ndarray
        Current node activities x_1..x_N, each with |x_i| < B_i.
    x_in : float
        Constant upstream input x_0 in [-1, 1].
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0, one per node.
    B : np.ndarray
        Edge saturation parameters B_i > 1, one per node.
    phi : np.ndarray
        Edge bias parameters phi_i in [-1, 1], one per node.

    Returns
    -------
    dxdt : np.ndarray
        Array of length N with dx_i/dt.

    Raises
    ------
    ValueError
        If the parameter arrays differ in length or violate alpha > 0, B > 1, |phi| <= 1; if x has the wrong length or |x_i| >= B_i; or if x_in is outside [-1, 1].
    """
    return dxdt
```

### Step 3

simulate_cascade

Goal
----
Integrate the cascade from the uniform initial activity x_init under the sustained input x_in and return the sampled activity profiles at t_j = j dt for j = 0..J as a (J + 1, N) array. J is the first sample at which the terminal node has moved by more than threshold from x_init (the run stops there); if that never happens, stop at j = max_steps. Integrate accurately (relative tolerance 1e-9 or tighter) so that the sampled profiles are reproducible to about 1e-8.

```python
def simulate_cascade(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray", x_in: float,
                             x_init: float, dt: float, threshold: float, max_steps: int) -> "np.ndarray":
    """Integrate the cascade from the uniform initial activity x_init under the sustained input x_in and return the sampled activity profiles at t_j = j dt for j = 0..J as a (J + 1, N) array. J is the first sample at which the terminal node has moved by more than threshold from x_init (the run stops there); if that never happens, stop at j = max_steps. Integrate accurately (relative tolerance 1e-9 or tighter) so that the sampled profiles are reproducible to about 1e-8.

    Parameters
    ----------
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0.
    B : np.ndarray
        Edge saturation parameters B_i > 1.
    phi : np.ndarray
        Edge bias parameters phi_i in [-1, 1].
    x_in : float
        Constant upstream input in [-1, 1].
    x_init : float
        Uniform initial activity of every node, in [-1, 1].
    dt : float
        Sampling interval, positive.
    threshold : float
        Positive deviation of the terminal node from x_init that ends the run.
    max_steps : int
        Positive cap on the number of sampling intervals.

    Returns
    -------
    profiles : np.ndarray
        Array of shape (J + 1, N); row j is the profile at time j dt.

    Raises
    ------
    ValueError
        If the parameter arrays are invalid, x_in or x_init is outside [-1, 1], dt or threshold is not positive, or max_steps is not a positive integer.
    """
    return profiles
```

### Step 4

instantaneous_speed

Goal
----
Estimate the instantaneous propagation speed of the front between two consecutive sampled profiles separated by dt, in units of position per unit time, with the source's shape-matching definition of the instantaneous wave velocity. The node positions may be the node indices or any increasing rescaled coordinate; the upstream input counts as node 0 at position 0, and positions before it take the input value. Search all candidate speeds from zero up to the whole pathway length per dt and return the global optimum.

```python
def instantaneous_speed(profile_prev: "np.ndarray", profile_next: "np.ndarray",
                                positions: "np.ndarray", x_in: float, dt: float) -> float:
    """Estimate the instantaneous propagation speed of the front between two consecutive sampled profiles separated by dt, in units of position per unit time, with the source's shape-matching definition of the instantaneous wave velocity. The node positions may be the node indices or any increasing rescaled coordinate; the upstream input counts as node 0 at position 0, and positions before it take the input value. Search all candidate speeds from zero up to the whole pathway length per dt and return the global optimum.

    Parameters
    ----------
    profile_prev : np.ndarray
        Node activities at the earlier sample.
    profile_next : np.ndarray
        Node activities at the later sample.
    positions : np.ndarray
        Strictly increasing positive node positions (index or rescaled).
    x_in : float
        Constant upstream input in [-1, 1] (node 0 at position 0).
    dt : float
        Time between the two samples, positive.

    Returns
    -------
    c : float
        Instantaneous speed.

    Raises
    ------
    ValueError
        If the three arrays differ in length, the positions are not strictly increasing and positive, x_in is outside [-1, 1], or dt is not positive.
    """
    return c
```

### Step 5

intrinsic_edge_speed

Goal
----
Compute the intrinsic propagation speed c(alpha, B, phi) of an edge: the asymptotic speed, in nodes per unit time, at which the front launched by the constant input x_in into the uniform resting state x_init travels through an infinitely long uniform pathway in which every edge carries the parameters (alpha, B, phi); that is, the speed at which the front's profile translates without change of shape once the boundary transient has decayed. The returned value must be accurate to a relative error of rel_tol or better; how the infinite-pathway limit is reached to that accuracy is part of the task.

```python
def intrinsic_edge_speed(alpha: float, B: float, phi: float, x_in: float, x_init: float,
                                 rel_tol: float) -> float:
    """Compute the intrinsic propagation speed c(alpha, B, phi) of an edge: the asymptotic speed, in nodes per unit time, at which the front launched by the constant input x_in into the uniform resting state x_init travels through an infinitely long uniform pathway in which every edge carries the parameters (alpha, B, phi); that is, the speed at which the front's profile translates without change of shape once the boundary transient has decayed. The returned value must be accurate to a relative error of rel_tol or better; how the infinite-pathway limit is reached to that accuracy is part of the task.

    Parameters
    ----------
    alpha : float
        Edge timescale parameter, positive.
    B : float
        Edge saturation parameter, greater than 1.
    phi : float
        Edge bias parameter, with |phi| < 1 / B (bistable edge).
    x_in : float
        Constant upstream input, +1 or -1.
    x_init : float
        Uniform resting state of the pathway, +1 or -1, different from x_in.
    rel_tol : float
        Required relative accuracy of the returned speed, in (0, 1e-6].

    Returns
    -------
    c : float
        Asymptotic front speed.

    Raises
    ------
    ValueError
        If alpha is not positive, B is not greater than 1, |phi| >= 1 / B, x_in or x_init is not +1 or -1, x_in equals x_init, rel_tol is outside (0, 1e-6], or the requested accuracy cannot be certified.
    """
    return c
```

### Step 6

rescaled_positions

Goal
----
Return the rescaled node coordinate s_1..s_N of the source for a pathway whose edges have the given intrinsic speeds (edge_speeds[i-1] is the speed of the edge feeding node i), constructed so that the spacing between consecutive nodes is inversely proportional to the intrinsic speed of the connecting edge and the whole rescaled pathway has unit length, with node 0 at s = 0.

```python
def rescaled_positions(edge_speeds: "np.ndarray") -> "np.ndarray":
    """Return the rescaled node coordinate s_1..s_N of the source for a pathway whose edges have the given intrinsic speeds (edge_speeds[i-1] is the speed of the edge feeding node i), constructed so that the spacing between consecutive nodes is inversely proportional to the intrinsic speed of the connecting edge and the whole rescaled pathway has unit length, with node 0 at s = 0.

    Parameters
    ----------
    edge_speeds : np.ndarray
        Positive intrinsic speeds, one per edge.

    Returns
    -------
    s : np.ndarray
        Rescaled node positions of length N.

    Raises
    ------
    ValueError
        If edge_speeds is empty, non-finite or contains a non-positive entry.
    """
    return s
```

### Step 7

wave_centre

Goal
----
Return the position of the wave centre of a sampled profile: the position at which the profile, interpolated piecewise-linearly between consecutive nodes with the upstream input counted as node 0 at position 0, first crosses zero coming from the input side. Return the terminal position if the profile never crosses zero.

```python
def wave_centre(positions: "np.ndarray", profile: "np.ndarray", x_in: float) -> float:
    """Return the position of the wave centre of a sampled profile: the position at which the profile, interpolated piecewise-linearly between consecutive nodes with the upstream input counted as node 0 at position 0, first crosses zero coming from the input side. Return the terminal position if the profile never crosses zero.

    Parameters
    ----------
    positions : np.ndarray
        Strictly increasing positive node positions.
    profile : np.ndarray
        Node activities at one sample.
    x_in : float
        Constant upstream input in [-1, 1].

    Returns
    -------
    centre : float
        Wave-centre position.

    Raises
    ------
    ValueError
        If the arrays differ in length, the positions are not strictly increasing and positive, or x_in is outside [-1, 1].
    """
    return centre
```

### Step 8

velocity_ise

Goal
----
Compute the velocity integral square error of a speed series over the post-transient window: the trapezoidal integral over time of the squared deviation of the speed from its mean over the window. speeds[j] is the speed on the sampling interval that starts at times[j]; the window consists of the intervals whose starting sample has the wave centre, given as a fraction of the pathway length in centres[j], in [s_lo, s_hi). The mean is the plain average of the speeds in the window.

```python
def velocity_ise(speeds: "np.ndarray", times: "np.ndarray", centres: "np.ndarray",
                         s_lo: float, s_hi: float) -> float:
    """Compute the velocity integral square error of a speed series over the post-transient window: the trapezoidal integral over time of the squared deviation of the speed from its mean over the window. speeds[j] is the speed on the sampling interval that starts at times[j]; the window consists of the intervals whose starting sample has the wave centre, given as a fraction of the pathway length in centres[j], in [s_lo, s_hi). The mean is the plain average of the speeds in the window.

    Parameters
    ----------
    speeds : np.ndarray
        Speed on each sampling interval.
    times : np.ndarray
        Strictly increasing start time of each sampling interval.
    centres : np.ndarray
        Wave-centre position at each start time, as a fraction of the pathway length.
    s_lo : float
        Lower window bound in [0, 1] (inclusive).
    s_hi : float
        Upper window bound in [0, 1] (exclusive), greater than s_lo.

    Returns
    -------
    vise : float
        Velocity integral square error.

    Raises
    ------
    ValueError
        If the arrays differ in length, the times do not increase, the bounds are invalid, or fewer than two intervals fall in the window.
    """
    return vise
```

### Step 9

fluctuation_suppression_ratio

Goal
----
Orchestrate the whole pipeline for a heterogeneous pathway: check that every edge is bistable and that the uniform initial activity is a resting state, simulate the cascade, obtain the intrinsic edge speeds (the intrinsic speed of an edge with timescale alpha_i is alpha_i times the intrinsic speed of the unit-timescale edge with the same B_i and phi_i), build the rescaled coordinate, estimate the instantaneous speed on every sampling interval in both the node-index coordinate and the rescaled coordinate, express both as fractions of the respective pathway length per unit time, locate the wave centre at every sample in the rescaled coordinate, and return the ratio of the velocity integral square error in the rescaled coordinate to that in the node-index coordinate over the post-transient window. Call the earlier step functions rather than reimplementing them.

```python
def fluctuation_suppression_ratio(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray",
                                          x_in: float, x_init: float, dt: float, threshold: float,
                                          speed_rel_tol: float, s_lo: float, s_hi: float) -> float:
    """Orchestrate the whole pipeline for a heterogeneous pathway: check that every edge is bistable and that the uniform initial activity is a resting state, simulate the cascade, obtain the intrinsic edge speeds (the intrinsic speed of an edge with timescale alpha_i is alpha_i times the intrinsic speed of the unit-timescale edge with the same B_i and phi_i), build the rescaled coordinate, estimate the instantaneous speed on every sampling interval in both the node-index coordinate and the rescaled coordinate, express both as fractions of the respective pathway length per unit time, locate the wave centre at every sample in the rescaled coordinate, and return the ratio of the velocity integral square error in the rescaled coordinate to that in the node-index coordinate over the post-transient window. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    alpha : np.ndarray
        Edge timescale parameters alpha_i > 0.
    B : np.ndarray
        Edge saturation parameters B_i > 1.
    phi : np.ndarray
        Edge bias parameters, each with |phi_i| < 1 / B_i.
    x_in : float
        Constant upstream input in [-1, 1].
    x_init : float
        Uniform initial activity in [-1, 1].
    dt : float
        Sampling interval, positive.
    threshold : float
        Positive terminal-node deviation that ends every run.
    speed_rel_tol : float
        Required relative accuracy of every intrinsic edge speed, in (0, 1e-6].
    s_lo : float
        Lower window bound in [0, 1].
    s_hi : float
        Upper window bound in [0, 1], greater than s_lo.

    Returns
    -------
    ratio : float
        VISE(rescaled) / VISE(node index).

    Raises
    ------
    ValueError
        If the parameters are invalid, an edge is not bistable, the uniform initial activity is not a resting state of the pathway, an intrinsic speed cannot be certified to speed_rel_tol, or the window contains fewer than two intervals.
    """
    return ratio
```
