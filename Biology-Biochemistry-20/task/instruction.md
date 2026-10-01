# Biology-Biochemistry-20

## Background

Cellular identity is held by a gene regulatory network whose many activating and repressing
interactions behave, once the number of genes is large, like a random matrix, and by chromatin
modifications that change on a far slower timescale and shift how readily each gene is switched
on. Because those marks respond to expression while expression responds to the marks, the
landscape on which a cell's state moves is itself remodelled as the cell develops, which is the
formal content of Waddington's epigenetic landscape. Whether a differentiated state is held
firmly or instead sits close to the edge of instability therefore depends on how strongly the
slow feedback reshapes the collective dynamics, and the same question underlies reprogramming,
where a state that was stable has to be destabilized. Networks of this size cannot be
characterized by simulating single realizations, because the quantity of interest is a property
of the disorder ensemble rather than of one wiring diagram, so a mean-field reduction that
survives the two-timescale structure is needed.

## Problem

I want to model a cell as N genes. This value of N tends to infinity, whose expression levels obey
`dx_i/dt = tanh(beta * [sum_j J_ij x_j + theta_i + c_i]) - x_i`, where the regulatory couplings
`J_ij` are drawn independently from a zero-mean Gaussian of variance `g^2/N` at unit disorder
strength `g = 1`, the response gain is `beta = 4`, and `c_i` is a constant external input. Every
gene also carries a chromatin mark that shifts its own activation threshold and relaxes far more
slowly than expression, `dtheta_i/dt = nu * [alpha * tanh(beta * (theta_i + c_i)) - theta_i]`
with `nu = 0.1` and feedback strength `alpha = 0.5`.

My sample population is sorted into five classes by external input, `c = -0.32, -0.10, +0.04, +0.11,
+0.35`, carrying population fractions `0.14, 0.22, 0.27, 0.21, 0.16`. In every class the marks
start at the 1000 equally spaced values running from `-5/2` to `+5/2` inclusive, and I follow the
system to frequence far beyond `1/nu`.

At those frequencies, I want the largest Lyapunov exponent
`lambda = lim_{tau -> inf} ln[chi^2(tau)] / (2 tau)` of the expression dynamics, where
`chi^2(tau) = lim_{s -> inf} N^{-1} sum_{i,j} < [d y_i(s + tau) / d u_j(s)]^2 >` is the coupling-
and population-averaged mean-square linear response of the total regulatory inputs
`y_i = sum_j J_ij x_j + theta_i + c_i` to infinitesimal independent impulses `u_j` added to their
equations of motion in the distant past. Report `lambda` to at least six significant figures.

In `<reasoning>`, state: how many distinct long-time mark values the whole population carries,
counting each class's attractors separately; whether the long-time expression dynamics come to
rest or keep fluctuating, and the evidence; the equal-time value and the infinite-lag limit of the
autocorrelation of the disorder-induced input `sum_j J_ij x_j`, each to at least four significant
figures; the lag at which that autocorrelation has fallen halfway from its equal-time value to its
limit, and the exponential rate at which it approaches the limit at long lags, each to at least
three significant figures; and whether the chromatin feedback leaves the long-time dynamics more or
less chaotic than the same population with `alpha = 0`, with the physical reason. Those are the
derived quantities that fix the final number or place it against the feedback-free case, so
stating them is what the output requirements below call for; what those requirements exclude is
restating the supplied inputs and per-gene mark tables, which this problem does not need.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_solve_epigenetic_equilibria

Goal
----
Locate every equilibrium of a gene class's slow chromatin-mark flow and report the linear rate at which the flow grows or decays about each one.

```python
def solve_epigenetic_equilibria(alpha: float, beta: float, c: float,
                                scan_points: int = 4001) -> "np.ndarray":
    """Return the equilibria of the slow mark flow together with their linear rates.

    The equilibria are the solutions of ``alpha * tanh(beta * (theta + c)) =
    theta``. The residual ``alpha * tanh(beta * (theta + c)) - theta`` is
    sampled at ``scan_points`` uniformly spaced locations over the interval
    ``[-(alpha + 1), alpha + 1]``, which always contains all equilibria. When
    the residual is non-monotone, its analytic stationary points are added to
    the scan so that close root pairs and double roots at folds are retained.
    Each sign-changing bracket is refined to double precision with Brent's
    method, and a sampled zero is taken as an equilibrium directly. Two
    equilibria closer together than ``1e-10`` are reported once.

    Row ``k`` of the result holds ``[theta_k, rate_k]``, where ``rate_k`` is the
    derivative ``alpha * beta / cosh(beta * (theta_k + c))**2 - 1`` of the
    residual at that equilibrium, so a negative rate marks an attractor and a
    positive one a repelling watershed. Rows are ordered by increasing
    ``theta_k``.

    Parameters
    ----------
    alpha : float
        Nonnegative finite feedback strength.
    beta : float
        Positive finite response gain.
    c : float
        Finite external input shared by the whole class.
    scan_points : int
        Number of bracketing samples, at least 3 (booleans are rejected).

    Returns
    -------
    np.ndarray
        Float array of shape ``(M, 2)`` with ``M >= 1`` equilibria.

    Raises
    ------
    ValueError
        If ``alpha`` is negative or not finite, if ``beta`` is not positive and
        finite, if ``c`` is not finite, if ``scan_points`` is not an integer of
        at least 3, or if the scan brackets no equilibrium.
    """
    return table
```

### Step 2

02_settle_starting_marks

Goal
----
Sort the starting chromatin marks of a gene class into the basins of its slow flow and return the constant regulatory input each occupied attractor carries together with the fraction of the class that settles there.

```python
def settle_starting_marks(starting: "np.ndarray", equilibria: "np.ndarray", c: float) -> "np.ndarray":
    """Return the occupied constant inputs of a class and their population shares.

    ``equilibria`` lists rows ``[theta, rate]`` ordered by increasing ``theta``.
    Away from a saddle-node fold, attractors (rate below ``-1e-12``) and
    repelling watersheds (rate above ``1e-12``) alternate, beginning and ending
    with an attractor. The basin of an attractor is the open interval between
    its flanking watersheds, unbounded beyond the outermost watersheds.

    A rate whose magnitude is at most ``1e-12`` is marginal. The two additional
    admissible tables produced at bifurcations are a single marginal equilibrium
    (the critical monostable state), or two equilibria consisting of one
    attractor and one marginal fold. At a two-equilibrium fold, the marginal
    point attracts from the unbounded side: marks at or left of a left marginal
    point settle there, and marks at or right of a right marginal point settle
    there. Marks on the side between the fold and the ordinary attractor settle
    at that attractor. A mark exactly on a positive-rate watershed remains
    invalid.

    Row ``k`` of the result holds ``[input_k, share_k]``, where ``input_k`` is a
    terminal equilibrium's mark plus ``c`` and ``share_k`` is the fraction of
    starting marks that settle there. Unoccupied terminal equilibria are
    omitted, rows are ordered by increasing input, and shares sum to one.

    Parameters
    ----------
    starting : np.ndarray
        Non-empty one-dimensional array of finite starting marks.
    equilibria : np.ndarray
        Finite ``(M, 2)`` array in one of the equilibrium patterns described
        above, with rows ordered by strictly increasing mark.
    c : float
        Finite external input shared by the whole class.

    Returns
    -------
    np.ndarray
        Float array of shape ``(J, 2)`` with one row per occupied terminal
        equilibrium.

    Raises
    ------
    ValueError
        If ``starting`` is not a non-empty one-dimensional array of finite
        numbers, if ``c`` is not finite, if ``equilibria`` is not a finite
        ``(M, 2)`` array with strictly increasing marks and one of the stated
        sign patterns, or if a starting mark coincides with a positive-rate
        watershed.
    """
    return table
```

### Step 3

03_average_correlation_overlap

Goal
----
Evaluate, at a set of correlation lag values, the population-weighted three-Gaussian pair average of the saturating response for an ensemble of constant regulatory inputs.

```python
def average_correlation_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                                weights: "np.ndarray", beta: float,
                                nodes: int = 241) -> "np.ndarray":
    """Return the population-weighted pair average of the saturating response at each lag value.

    For each entry ``d`` of ``deltas``, let ``a = sqrt(abs(d))``,
    ``b = sqrt(delta_zero - abs(d))`` and ``sigma = +1`` when ``d >= 0`` or
    ``-1`` otherwise. With ``z1``, ``z2`` and ``z3`` independent standard
    normal variables, entry ``i`` of the result is

    ``sum_j weights[j] * E[ tanh(beta * (b*z1 + a*z3 + fields[j]))
                          * tanh(beta * (b*z2 + sigma*a*z3 + fields[j])) ]``.

    Thus the two Gaussian arguments each have variance ``delta_zero`` and
    covariance ``d``, including anticorrelated lags. At ``d = 0`` the shared
    term vanishes, so the choice ``sigma = +1`` is immaterial.

    Every standard-normal expectation ``E[f(z)]`` is evaluated as
    ``sum_k q_k f(x_k)`` on the ``nodes`` equally spaced abscissae
    ``x_k = -12 + 24 k / (nodes - 1)`` with weights
    ``q_k = (24 / (nodes - 1)) * exp(-x_k**2 / 2) / sqrt(2 pi)``, in each of the
    three variables. Only lag values with ``abs(d) <= delta_zero`` are admissible.

    Parameters
    ----------
    deltas : np.ndarray
        Non-empty one-dimensional array of finite lag values.
    delta_zero : float
        Nonnegative finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3 (booleans
        are rejected).

    Returns
    -------
    np.ndarray
        Float array of the same length as ``deltas``.

    Raises
    ------
    ValueError
        If ``deltas`` is not a non-empty one-dimensional array of finite numbers
        or holds a value outside ``[-delta_zero, delta_zero]``, if ``delta_zero`` is
        negative or not finite, if ``fields`` is not a non-empty one-dimensional
        array of finite numbers, if ``weights`` does not match ``fields`` in
        length or is negative or does not sum to one within ``1e-9``, if
        ``beta`` is not positive and finite, or if ``nodes`` is not an integer of
        at least 3.
    """
    return overlaps
```

### Step 4

04_average_gain_overlap

Goal
----
Evaluate, at a set of correlation lag values, the population-weighted three-Gaussian pair average of the response slope for an ensemble of constant regulatory inputs.

```python
def average_gain_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                         weights: "np.ndarray", beta: float, nodes: int = 241) -> "np.ndarray":
    """Return the population-weighted pair average of the response slope at each lag value.

    For each entry ``d`` of ``deltas``, use the signed-covariance construction
    from ``average_correlation_overlap``: ``a = sqrt(abs(d))``,
    ``b = sqrt(delta_zero - abs(d))`` and ``sigma = sign(d)`` with
    ``sigma = +1`` at zero. Entry ``i`` of the result is

    ``sum_j weights[j] * E[ s(b*z1 + a*z3 + fields[j])
                          * s(b*z2 + sigma*a*z3 + fields[j]) ]``

    with the slope ``s(x) = beta / cosh(beta * x)**2`` of ``tanh(beta * x)``.
    Expectations use the same equally spaced standard-normal rule as
    ``average_correlation_overlap``: abscissae ``x_k = -12 + 24 k / (nodes - 1)``
    and weights ``q_k = (24 / (nodes - 1)) * exp(-x_k**2 / 2) / sqrt(2 pi)`` in
    each variable. Only lag values with ``abs(d) <= delta_zero`` are admissible.

    Parameters
    ----------
    deltas : np.ndarray
        Non-empty one-dimensional array of finite lag values.
    delta_zero : float
        Nonnegative finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3 (booleans
        are rejected).

    Returns
    -------
    np.ndarray
        Nonnegative float array of the same length as ``deltas``.

    Raises
    ------
    ValueError
        Under the same conditions as ``average_correlation_overlap``.
    """
    return overlaps
```

### Step 5

05_average_potential_overlap

Goal
----
Evaluate, at a set of correlation lag values, the population-weighted three-Gaussian pair average of the log-cosh response integral for an ensemble of constant regulatory inputs.

```python
def average_potential_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                              weights: "np.ndarray", beta: float,
                              nodes: int = 241) -> "np.ndarray":
    """Return the population-weighted pair average of the log-cosh integral at each lag value.

    For each entry ``d`` of ``deltas``, use the signed-covariance construction
    from ``average_correlation_overlap``: ``a = sqrt(abs(d))``,
    ``b = sqrt(delta_zero - abs(d))`` and ``sigma = sign(d)`` with
    ``sigma = +1`` at zero. Entry ``i`` of the result is

    ``sum_j weights[j] * E[ P(b*z1 + a*z3 + fields[j])
                          * P(b*z2 + sigma*a*z3 + fields[j]) ]``

    with ``P(x) = ln(cosh(beta * x)) / beta``, evaluated without overflow for
    large arguments. Expectations use the same equally spaced standard-normal
    rule as ``average_correlation_overlap``. Only lag values with
    ``abs(d) <= delta_zero`` are admissible.

    Parameters
    ----------
    deltas : np.ndarray
        Non-empty one-dimensional array of finite lag values.
    delta_zero : float
        Nonnegative finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3 (booleans
        are rejected).

    Returns
    -------
    np.ndarray
        Nonnegative float array of the same length as ``deltas``.

    Raises
    ------
    ValueError
        Under the same conditions as ``average_correlation_overlap``.
    """
    return overlaps
```

### Step 6

06_solve_frozen_correlation

Goal
----
Solve the self-consistency condition of a time-independent collective state for the equal-time correlation of the disorder-induced part of a gene's regulatory input.

```python
def solve_frozen_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                             lower: float, upper: float, nodes: int = 241) -> float:
    """Return the equal-time correlation of a time-independent collective state.

    The state is time-independent when the equal-time residual

    ``average_correlation_overlap(np.array([d]), d, fields, weights, beta, nodes)[0] - d``

    vanishes. The root is located in ``[lower, upper]`` with Brent's method to
    double precision. The residual must not take the same non-zero sign at the
    two ends; if it is exactly zero at an end, that end is returned, and
    ``lower`` wins when it is exactly zero at both.

    Parameters
    ----------
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    lower : float
        Nonnegative finite lower end of the search bracket.
    upper : float
        Finite upper end of the search bracket, greater than ``lower``.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.

    Returns
    -------
    float
        The self-consistent equal-time correlation, a native Python float.

    Raises
    ------
    ValueError
        If ``lower`` is negative or not finite, if ``upper`` is not finite or not
        greater than ``lower``, if the residual has the same non-zero sign at
        both ends, or if ``fields``, ``weights``, ``beta`` or ``nodes`` fails the
        contract of ``average_correlation_overlap``.
    """
    return correlation
```

### Step 7

07_locate_correlation_hilltop

Goal
----
Find the lowest interior maximum of the effective potential that governs the autocorrelation of the disorder-induced input, for a given equal-time correlation.

```python
def locate_correlation_hilltop(delta_zero: float, fields: "np.ndarray", weights: "np.ndarray",
                               beta: float, nodes: int = 241, scan_points: int = 17) -> float:
    """Return the lowest lag value at which the potential slope turns from rising to falling.

    The slope of the effective potential at lag value ``d`` is

    ``slope(d) = average_correlation_overlap(np.array([d]), delta_zero, fields, weights, beta, nodes)[0] - d``.

    It is sampled at ``scan_points`` equally spaced lag values from ``0`` to
    ``delta_zero`` inclusive. For the first consecutive pair of samples with
    ``slope > 0`` followed by ``slope <= 0``, the root inside that pair is refined
    to double precision with Brent's method and returned (a later sample whose
    slope is exactly zero is itself that root). If no such pair exists the
    potential has no interior hilltop and ``nan`` is returned.

    Parameters
    ----------
    delta_zero : float
        Positive finite equal-time correlation.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    scan_points : int
        Number of equally spaced scan samples, at least 3 (booleans are rejected).

    Returns
    -------
    float
        The hilltop lag value, or ``nan`` when there is none.

    Raises
    ------
    ValueError
        If ``delta_zero`` is not positive and finite, if ``scan_points`` is not an
        integer of at least 3, or if ``fields``, ``weights``, ``beta`` or ``nodes``
        fails the contract of ``average_correlation_overlap``.
    """
    return hilltop
```

### Step 8

08_solve_decaying_correlation

Goal
----
Solve for the equal-time correlation and the infinite-lag limit of a decaying autocorrelation, the solution that starts at rest and climbs asymptotically onto a hilltop of the effective potential.

```python
def solve_decaying_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                               lower: float, upper: float, nodes: int = 241,
                               scan_points: int = 17) -> "np.ndarray":
    """Return the equal-time correlation and infinite-lag limit of the decaying autocorrelation.

    For an equal-time correlation ``d0`` the effective potential at lag value
    ``d`` is ``V(d) = average_potential_overlap(np.array([d]), d0, ...)[0] - d**2 / 2``.
    With ``h = locate_correlation_hilltop(d0, fields, weights, beta, nodes, scan_points)``
    the energy gap is ``V(d0) - V(h)``; when ``h`` is ``nan`` (no hilltop) the gap is
    taken as ``+1``. The gap must be positive at ``lower`` and negative at
    ``upper``; its sign change is located with Brent's method to double
    precision, giving ``d0``. The result is ``[d0, h(d0)]``.

    Parameters
    ----------
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    lower : float
        Positive finite lower end of the search bracket.
    upper : float
        Finite upper end of the search bracket, greater than ``lower``.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    scan_points : int
        Number of hilltop scan samples, at least 3.

    Returns
    -------
    np.ndarray
        Float array ``[equal_time_correlation, infinite_lag_limit]``.

    Raises
    ------
    ValueError
        If ``lower`` is not positive and finite, if ``upper`` is not finite or not
        greater than ``lower``, if the gap is not positive at ``lower`` and
        negative at ``upper``, or if an argument fails the contract of
        ``locate_correlation_hilltop`` or ``average_potential_overlap``.
    """
    return solution
```

### Step 9

09_trace_decay_trajectory

Goal
----
Trace the decaying autocorrelation of the disorder-induced input as a function of lag time, from its equal-time value down onto the hilltop of the effective potential.

```python
def trace_decay_trajectory(delta_zero: float, delta_inf: float, fields: "np.ndarray",
                           weights: "np.ndarray", beta: float, t_max: float, n_tau: int,
                           nodes: int = 241, degree: int = 64) -> "np.ndarray":
    """Return the decaying autocorrelation sampled on a uniform grid of lag times.

    The acceleration of the autocorrelation ``D`` is ``-slope(D)`` with
    ``slope(D) = average_correlation_overlap(np.array([D]), delta_zero, ...)[0] - D``.
    Represent this slope by its degree-``degree`` interpolant at the first-kind
    Chebyshev nodes on ``[delta_inf, delta_zero]``. Let ``top`` be the root of
    that interpolant in the interval extending ``0.001 * (delta_zero-delta_inf)``
    to either side of ``delta_inf``.

    Return the unique physical branch of ``D'' = -slope(D)`` that is even in lag
    (so ``D'(0)=0``), decreases for positive lag, and approaches ``top`` along
    its decaying exponential mode as lag tends to infinity. Its turning point
    must lie above the midpoint between ``delta_inf`` and ``delta_zero``. Entry
    ``k`` is this solution at ``tau_k = k * t_max / (n_tau - 1)``. Any stable
    numerical method that satisfies these boundary conditions may be used.

    Parameters
    ----------
    delta_zero : float
        Finite equal-time correlation, greater than ``delta_inf``.
    delta_inf : float
        Nonnegative finite hilltop lag value of the effective potential.
    fields : np.ndarray
        Non-empty one-dimensional array of finite constant regulatory inputs.
    weights : np.ndarray
        Nonnegative population shares of the same length as ``fields``, summing
        to one within ``1e-9``.
    beta : float
        Positive finite response gain.
    t_max : float
        Positive finite largest lag time of the grid.
    n_tau : int
        Number of grid points, at least 2.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    degree : int
        Chebyshev interpolation degree, at least 4.

    Returns
    -------
    np.ndarray
        Float array of length ``n_tau``, starting at the turning point near
        ``delta_zero`` and decreasing towards ``delta_inf``.

    Raises
    ------
    ValueError
        If ``delta_inf`` is negative or not finite, if ``delta_zero`` is not
        finite or not greater than ``delta_inf``, if ``t_max`` is not positive and
        finite, if ``n_tau`` or ``degree`` is too small, if the interpolated slope
        does not fall from positive to negative across the bracket around
        ``delta_inf`` or ``1 -`` the gain overlap at its root is not positive (no
        hilltop there), if the integration never comes to rest or comes to rest
        below the midpoint of ``[delta_inf, delta_zero]``, or if an argument fails
        the contract of the overlap steps.
    """
    return profile
```

### Step 10

10_solve_fluctuation_ground_state

Goal
----
Step name: Return the bound-state ground energy of the one-dimensional fluctuation operator for an even-lag potential, matching the sampled well to its constant long-lag plateau.

```python
def solve_fluctuation_ground_state(half_profile: "np.ndarray", spacing: float) -> float:
    """Return the even bound-state ground energy of the fluctuation operator.

    Entry ``k`` of ``half_profile`` is the fluctuation potential ``W`` at lag
    time ``k * spacing`` for ``k = 0 .. n - 1``. Between samples, use the
    continuous piecewise-linear potential; beyond the last sample the potential
    is the constant plateau ``W[-1]``. On the half-line, solve

    ``-psi''(tau) + W(tau) * psi(tau) = E * psi(tau)``

    for the smallest energy ``E < W[-1]`` having an even solution
    (``psi'(0)=0``) that decays at infinity. At the last sampled time ``T``, the
    exact constant-tail condition is
    ``psi'(T) + sqrt(W[-1] - E) * psi(T) = 0``. These two boundary conditions make
    ``E`` a nonlinear bound-state root. The normalization of ``psi`` is
    arbitrary, and the supplied array must not be modified. Any converged
    shooting, boundary-value, transfer-matrix, or equivalent method is valid.

    Parameters
    ----------
    half_profile : np.ndarray
        One-dimensional array of at least 3 finite samples on ``tau >= 0``.
    spacing : float
        Positive finite lag-time spacing of the samples.

    Returns
    -------
    float
        The lowest even bound-state energy, as a native Python float.

    Raises
    ------
    ValueError
        If ``half_profile`` is not a one-dimensional array of at least 3 finite
        numbers, if ``spacing`` is not positive and finite, or if the sampled
        potential admits no even bound state below its long-lag plateau.
    RuntimeError
        If the numerical boundary-value solve does not converge to a nodeless
        bound state.
    """
    return ground_energy
```

### Step 11

11_compute_settled_lyapunov_exponent

Goal
----
Compose every earlier step to obtain the largest Lyapunov exponent of the long-time collective state of a heterogeneous gene population whose chromatin marks have settled.

```python
def compute_settled_lyapunov_exponent(
    inputs: tuple = (-0.32, -0.1, 0.04, 0.11, 0.35),
    fractions: tuple = (0.14, 0.22, 0.27, 0.21, 0.16),
    alpha: float = 0.5,
    beta: float = 4.0,
    n_marks: int = 1000,
    mark_span: float = 2.5,
    nodes: int = 241,
    t_max: float = 40.0,
    n_tau: int = 4001,
) -> float:
    """Return the largest Lyapunov exponent of the settled collective state.

    Class ``k`` has external input ``inputs[k]`` and population fraction
    ``fractions[k]``; its marks start at the ``n_marks`` equally spaced values
    from ``-mark_span`` to ``+mark_span`` inclusive and settle onto the
    attractors of the class's slow flow. Each occupied attractor contributes one
    constant input weighted by the class fraction times its basin share, and the
    weights are rescaled to sum to one exactly.

    Select the largest physical time-independent self-consistency root in
    ``[0, 1]`` and test it with the fluctuation potential
    ``W = 1 - average_gain_overlap([d_star], d_star, ...)``. If ``W >= 0`` the
    root is stable and its constant-potential ground energy gives the exponent.
    Otherwise use the earlier solvers to find the nonconstant autocorrelation:
    its long-lag value must independently be a hilltop returned by
    ``locate_correlation_hilltop`` and its effective potential, evaluated with
    ``average_potential_overlap``, must equal that at the release point.

    Trace this physical branch on the nested uniform lag grids containing
    ``n_tau`` and ``2 * n_tau - 1`` points over ``[0, t_max]``. On each grid form
    ``1 - average_gain_overlap`` along the trajectory and solve the tail-matched
    continuous fluctuation problem. Remove the leading second-order error of
    the piecewise-linear potential using ``(4 * E_fine - E_coarse) / 3`` with the
    two nested energies before applying ``lambda = -1 + sqrt(1 - E0)``. The
    defaults reproduce the problem statement. Numerical bracketing and
    interpolation choices must be inferred from the contracts of the earlier
    steps rather than duplicated here.

    Parameters
    ----------
    inputs : tuple
        Non-empty sequence of finite external inputs, one per gene class.
    fractions : tuple
        Positive population fractions of the same length as ``inputs``, summing
        to one within ``1e-9``.
    alpha : float
        Nonnegative finite feedback strength.
    beta : float
        Positive finite response gain.
    n_marks : int
        Number of starting marks per class, at least 2 (booleans are rejected).
    mark_span : float
        Positive finite half-width of the starting mark interval.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    t_max : float
        Positive finite length of the lag-time window.
    n_tau : int
        Number of lag-time grid points of the coarser trajectory, at least 3.

    Returns
    -------
    float
        The largest Lyapunov exponent, a native Python float.

    Raises
    ------
    ValueError
        If ``inputs`` is not a non-empty one-dimensional sequence of finite
        numbers, if ``fractions`` does not match ``inputs`` in length or is not
        positive or does not sum to one within ``1e-9``, if ``n_marks`` or
        ``n_tau`` is too small, if ``mark_span`` or ``t_max`` is not positive and
        finite, if no time-independent state is found on ``[0, 1]``, if an
        unstable time-independent state admits no decaying autocorrelation in its
        bracket, if the lowest eigenvalue exceeds one, or if an argument fails the
        contract of the step that consumes it.
    RuntimeError
        If the continuous fluctuation eigenproblem does not converge to a
        nodeless bound state.
    """
    return 0.0
```
