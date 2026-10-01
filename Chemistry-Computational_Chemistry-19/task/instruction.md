# Chemistry-Computational_Chemistry-19

## Background

Single-molecule force spectroscopy shows that the force at which a polymer chain ruptures is not a fixed bond strength but a stochastic, loading-rate-dependent quantity governed by thermally activated bond dissociation. Kinetic descriptions of chain scission usually assume that the chain is pulled by a prescribed force, which makes scission irreversible, yet self-healing elastomers, vitrimers and networks built from dynamic or sacrificial bonds show that broken bonds can re-form. When the end-to-end distance of a chain is prescribed instead of the force, whether and how often a broken bond re-forms can change the rupture statistics that such experiments record.

## Problem

I pull single supramolecular polymer chains whose repeat units are joined by identical reversible bonds, so a chain can break at any one of its segments and the broken bond can re-form while the ends stay clamped. I model a molecule as N freely jointed Kuhn segments in three dimensions with length l = 0.400 nm: the breaking segment has a 12-6 Lennard-Jones energy of well depth ε0 with its minimum at length l, and the other N − 1 segments are rigid, their combined end-to-end vector of length r carrying the Kuhn–Grün free energy (N − 1) k_BT [F coth F + ln(F / sinh F)], where F = e(3 − e²)/(1 − e²) is the rounded Padé approximant of the inverse Langevin function at e = r/((N − 1) l). Scission and re-formation are thermally activated over the free-energy barriers along the length of the breaking segment at prescribed end-to-end distance, both with attempt frequency 1.0 × 10^13 s^-1, at T = 300.0 K.

I clamped one molecule at an end-to-end distance of 11.60 nm and recorded a telegraph signal with a mean intact dwell time of 57.70 ms and a mean broken dwell time of 52.23 ms; the molecule came from a polydisperse batch known to contain 45 through 60 segments per chain, so I know neither its exact N nor its ε0. I then pulled the same molecule from zero extension at a constant 1.000 nm/s under displacement control, starting intact.

What is the mean tension at the chain's final scission, the last one after which it never re-forms, averaged over repeated pulls and reported in piconewtons, where the tension at a given end-to-end distance is the derivative, with respect to that distance, of the chain's free energy at its intact-state minimum? Alongside the number, report the inferred segment count and bond well depth, quantify the segment-count sensitivity using adjacent allowed counts, give the equilibrium intact probability at the clamp, and predict the mean number of scission events per pull at 1.000 nm/s. Also report the mean tension obtained by pooling all scission events across repeated pulls with equal weight per event, and the mean final-scission tension at 0.100 nm/s, comparing each with the 1.000 nm/s final-scission mean through an absolute or percentage difference.

Output Format Requirements:
Emit <final_answer>...</final_answer> first, followed by <reasoning>...</reasoning>.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep the reasoning concise, but include the free-energy and kinetic/statistical definitions supporting the predictions, the requested intermediate results, and the numerical thermal force scale used to convert tension to piconewtons. Compact equations are welcome; a bare list of final numbers is insufficient. State numerical forces to three decimal places and other computed quantities with enough precision to support the comparisons. Do not include full numerical grids or iteration logs.

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

01_compute_link_free_energy

Goal
----
Evaluate the free energy of a displacement-controlled chain with one breakable segment as a function of that segment's length.

```python
def compute_link_free_energy(x_bar: "np.ndarray", y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Return the free energy profile along the length of the breakable segment.

    Lengths are in Kuhn lengths and energies in k_B T. The chain has
    ``n_segments`` freely jointed segments. One of them is breakable, with the
    12-6 Lennard-Jones energy ``bond_energy * (x**-12 - 2 * x**-6)`` of its
    length ``x``. The other ``m = n_segments - 1`` segments are rigid. Their
    combined end-to-end vector, which joins the breakable segment to the fixed
    end-to-end vector of length ``y_bar``, has length ``r`` and carries the
    Kuhn-Grun free energy ``m * (F / tanh(F) + ln(F / sinh(F)))`` with
    ``F = e * (3 - e**2) / (1 - e**2)`` and ``e = r / m``. The rigid segments
    cannot reach ``r >= m``. The scalar-length marginal includes the spherical
    Jacobian ``x_bar**2``, so the dimensionless free energy includes
    ``-2 * ln(x_bar)`` in addition to the Lennard-Jones and orientational terms.

    The returned array is the free energy of the chain as a function of the
    length ``x_bar`` of the breakable segment at fixed ``y_bar``. It is defined
    up to one additive constant that may depend on ``n_segments`` but not on
    ``x_bar``, ``y_bar`` or ``bond_energy``. Finite entries must be accurate to
    1e-9, including near full extension where the rigid-segment Boltzmann
    weight is smaller than the smallest positive double.

    Parameters
    ----------
    x_bar : np.ndarray
        One-dimensional array of positive breakable-segment lengths.
    y_bar : float
        Fixed end-to-end distance, at least 0.
    n_segments : int
        Number of Kuhn segments, at least 2.
    bond_energy : float
        Lennard-Jones well depth in k_B T, positive.

    Returns
    -------
    free_energy : np.ndarray
        Free energy for each entry of ``x_bar`` (same shape), equal to
        ``+inf`` where ``abs(y_bar - x_bar) >= n_segments - 1``.

    Raises
    ------
    ValueError
        If ``n_segments`` is not an integer of at least 2, if ``bond_energy``
        is not positive, if ``y_bar`` is negative, or if ``x_bar`` is not a
        one-dimensional array of positive values.
    """
    return free_energy
```

### Step 2

02_locate_free_energy_extrema

Goal
----
Locate the intact-state minimum, the transition-state maximum and the broken-state minimum of the free energy profile along the breakable segment's length at a fixed end-to-end distance.

```python
def locate_free_energy_extrema(y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Return the three stationary points that define scission and re-formation.

    The profile is ``compute_link_free_energy(x, y_bar, n_segments, bond_energy)``
    as a function of the breakable-segment length ``x`` (Kuhn lengths). The
    intact state is the local minimum nearest the bond length ``x = 1``, the
    transition state is the local maximum that immediately follows it at larger
    ``x``, and the broken state is the next local minimum after that maximum.
    If the required minimum-maximum-minimum pattern does not exist, every entry
    is NaN. Each reported location must be accurate to 1e-8.

    Parameters
    ----------
    y_bar : float
        Fixed end-to-end distance in Kuhn lengths, nonnegative and below
        ``n_segments + 1``.
    n_segments : int
        Number of Kuhn segments, at least 10.
    bond_energy : float
        Lennard-Jones well depth in k_B T, between 20 and 150.

    Returns
    -------
    locations : np.ndarray
        Array ``[x_intact, x_transition, x_broken]`` of shape (3,), or three
        NaN values when that sequence of stationary points does not exist.

    Raises
    ------
    ValueError
        If ``y_bar`` is negative or not below ``n_segments + 1``, if
        ``n_segments`` is not an integer of at least 10, or if ``bond_energy``
        lies outside [20, 150].
    """
    return locations
```

### Step 3

03_compute_intact_chain_tension

Goal
----
Compute the mean tension of the chain from the end-to-end distance derivative of its free energy at a fixed breakable-segment length.

```python
def compute_intact_chain_tension(y_bar: float, x_bar: float, n_segments: int) -> float:
    """Return the reduced tension at fixed breakable-segment length.

    The tension, in units of k_B T per Kuhn length, is the partial derivative
    with respect to ``y_bar`` of the free energy returned by
    ``compute_link_free_energy`` at the fixed segment length ``x_bar``.
    Evaluated at the intact-state minimum it is the mean tension of the intact
    chain. The bond energy does not enter. The result must have a relative
    accuracy of 1e-9.

    Parameters
    ----------
    y_bar : float
        End-to-end distance in Kuhn lengths, nonnegative, with
        ``abs(y_bar - x_bar)`` below ``n_segments - 1`` minus 0.01.
    x_bar : float
        Breakable-segment length in Kuhn lengths, positive.
    n_segments : int
        Number of Kuhn segments, at least 2.

    Returns
    -------
    tension : float
        Reduced tension as a native Python float.

    Raises
    ------
    ValueError
        If ``y_bar`` is negative or ``x_bar`` is not positive, if ``n_segments`` is not an
        integer of at least 2, or if ``abs(y_bar - x_bar)`` is not below
        ``n_segments - 1.01``.
    """
    return tension
```

### Step 4

04_infer_segment_count_and_bond_energy

Goal
----
Infer the number of Kuhn segments and the bond well depth of a clamped chain from its mean intact and broken dwell times at one end-to-end distance.

```python
def infer_segment_count_and_bond_energy(y_bar_hold: float, nu_tau_intact: float, nu_tau_broken: float, n_min: int, n_max: int) -> "np.ndarray":
    """Return the segment count and well depth that reproduce two dwell times.

    At the clamped end-to-end distance ``y_bar_hold`` (Kuhn lengths) the
    stationary points from ``locate_free_energy_extrema`` define the scission
    barrier (transition-state minus intact-state free energy) and the
    re-formation barrier (transition-state minus broken-state free energy),
    both in k_B T from ``compute_link_free_energy``. Scission can occur at any
    one of the ``n_segments`` equivalent segments, while a broken chain re-forms
    only through the bond that broke. Both elementary processes are Arrhenius
    with the same attempt frequency, and times are given in units of its
    inverse. The mean intact dwell time is the inverse of the total scission
    rate and the mean broken dwell time is the inverse of the re-formation rate.

    For each integer ``n_segments`` in ``[n_min, n_max]`` the well depth is the
    value in [20, 150] that reproduces ``nu_tau_intact`` exactly. The returned
    segment count is the one whose fitted well depth brings the natural log of
    the predicted broken dwell time closest to ``ln(nu_tau_broken)``, with ties
    resolved toward the smaller count. The inclusive search range contains at
    most 101 candidates and has ``n_max <= 10000``. Callers supply data for
    which every count in the range admits such a well depth. The well depth
    must be accurate to 1e-8.

    Parameters
    ----------
    y_bar_hold : float
        Clamped end-to-end distance in Kuhn lengths.
    nu_tau_intact : float
        Mean intact dwell time times the attempt frequency, positive.
    nu_tau_broken : float
        Mean broken dwell time times the attempt frequency, positive.
    n_min : int
        Smallest segment count considered, at least 10.
    n_max : int
        Largest segment count considered, at least ``n_min``, at most 10000,
        and no more than 100 above ``n_min``.

    Returns
    -------
    estimate : np.ndarray
        Array ``[n_segments, bond_energy]`` of shape (2,) and float dtype.

    Raises
    ------
    ValueError
        If either dwell time is not positive, if ``n_min`` is not an integer of
        at least 10, or if ``n_max`` is not an integer in
        ``[n_min, min(n_min + 100, 10000)]``.
    """
    return estimate
```

### Step 5

05_integrate_intact_probability

Goal
----
Propagate the probability that a chain pulled at a constant rate is intact, given its scission and re-formation barriers along the end-to-end distance.

```python
def integrate_intact_probability(y_grid: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Return the intact probability at every node of a pulling protocol.

    The end-to-end distance grows at the constant reduced rate ``loading_rate``
    (Kuhn lengths per inverse attempt frequency) through the nodes of
    ``y_grid``. An intact chain breaks with the total scission rate of its
    ``n_segments`` equivalent segments, and a broken chain re-forms with the
    re-formation rate of the bond that broke. Both rates are Arrhenius with unit
    attempt frequency over the barriers (k_B T) given at the nodes, and the
    chain is intact at the first node.

    Across each interval both rate coefficients are held at the geometric means
    of their two endpoint values, the equilibrium intact probability is taken
    to vary linearly between its values at the two endpoints (each computed from
    that endpoint's own rates), and the probability is advanced by the exact
    solution of the resulting linear equation. The update must remain exact when
    an interval spans many relaxation lengths.

    Parameters
    ----------
    y_grid : np.ndarray
        Strictly increasing one-dimensional grid with at least two nodes.
    scission_barrier : np.ndarray
        Scission barrier at each node, same shape as ``y_grid``.
    healing_barrier : np.ndarray
        Re-formation barrier at each node, same shape as ``y_grid``.
    n_segments : int
        Number of equivalent segments that can break, at least 1.
    loading_rate : float
        Reduced loading rate, positive.

    Returns
    -------
    intact_probability : np.ndarray
        Intact probability at each node, same shape as ``y_grid``.

    Raises
    ------
    ValueError
        If ``y_grid`` is not strictly increasing with at least two nodes, if the
        barrier arrays do not match its shape, if ``n_segments`` is not a
        positive integer, or if ``loading_rate`` is not positive.
    """
    return intact_probability
```

### Step 6

06_compute_final_scission_statistics

Goal
----
Compute the unnormalized first moments of tension and extension at the last scission event of a pulled reversible chain, together with the total final-scission probability and the expected number of scission events.

```python
def compute_final_scission_statistics(y_grid: "np.ndarray", intact_probability: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", tension: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Return statistics of the final (never re-formed) scission event.

    The chain is pulled at the constant reduced ``loading_rate`` through the
    nodes of ``y_grid``, with ``intact_probability`` from
    ``integrate_intact_probability`` and the same Arrhenius kinetics: the total
    scission rate of ``n_segments`` equivalent segments and the re-formation
    rate of the broken bond, over barriers in k_B T with unit attempt frequency.
    A scission at a node is final when the broken chain does not re-form at any
    later point of the grid; re-formation is impossible beyond the last node,
    where any probability still intact breaks as a point mass.

    Every integral over the grid, including the accumulated re-formation
    exposure between a node and the last node, uses the trapezoidal rule on the
    nodes, and the point mass at the last node enters all four results.

    Parameters
    ----------
    y_grid : np.ndarray
        Strictly increasing one-dimensional grid with at least two nodes.
    intact_probability : np.ndarray
        Intact probability at each node.
    scission_barrier : np.ndarray
        Scission barrier at each node.
    healing_barrier : np.ndarray
        Re-formation barrier at each node.
    tension : np.ndarray
        Reduced tension of the intact chain at each node.
    n_segments : int
        Number of equivalent segments that can break, at least 1.
    loading_rate : float
        Reduced loading rate, positive.

    Returns
    -------
    statistics : np.ndarray
        Shape (4,) array ``[unnormalized first moment of tension at the final
        scission, unnormalized first moment of end-to-end distance at the final
        scission, total probability of a final scission, expected number of
        scission events per pull]``. Divide either first moment by the returned
        total probability to obtain the corresponding conditional mean when
        that probability is not one.

    Raises
    ------
    ValueError
        If ``y_grid`` is not strictly increasing with at least two nodes, if any
        other array does not match its shape, if ``n_segments`` is not a
        positive integer, or if ``loading_rate`` is not positive.
    """
    return statistics
```

### Step 7

07_predict_mean_final_rupture_force

Goal
----
Predict the mean tension at the final scission of a reversible chain pulled at constant speed, starting from its clamped dwell times.

```python
def predict_mean_final_rupture_force(hold_extension_nm: float, intact_dwell_s: float, broken_dwell_s: float, pulling_speed_nm_s: float, kuhn_length_nm: float, temperature_k: float, attempt_frequency_hz: float, n_min: int, n_max: int) -> float:
    """Return the mean tension (pN) of the intact chain at its final scission.

    Convert the clamped end-to-end distance to Kuhn lengths, the dwell times to
    units of the inverse attempt frequency and the pulling speed to the reduced
    loading rate (Kuhn lengths per inverse attempt frequency). Infer the segment
    count within ``[n_min, n_max]`` and the well depth with
    ``infer_segment_count_and_bond_energy``.

    Tabulate the pull on the nodes ``0.0, 0.5, 1.0, 1.5, ...`` Kuhn lengths, stopping
    before the first node at which ``locate_free_energy_extrema`` finds no intact
    state. At each node take the scission and re-formation barriers from
    ``compute_link_free_energy`` at those stationary points and the tension from
    ``compute_intact_chain_tension`` at the intact-state length. Interpolate the
    two barriers and the tension with not-a-knot cubic splines through the
    nodes onto the uniform grid from zero to the last node with spacing 0.001,
    starting the chain intact at zero. Propagate the intact probability with
    ``integrate_intact_probability``, obtain the mean reduced tension at the
    final scission with ``compute_final_scission_statistics``, and multiply it
    by ``k_B * temperature_k / kuhn_length`` with k_B = 1.380649e-23 J/K,
    reporting piconewtons.

    Parameters
    ----------
    hold_extension_nm : float
        Clamped end-to-end distance of the dwell-time measurement (nm).
    intact_dwell_s : float
        Mean intact dwell time at that distance (s).
    broken_dwell_s : float
        Mean broken dwell time at that distance (s).
    pulling_speed_nm_s : float
        Constant pulling speed (nm/s), positive.
    kuhn_length_nm : float
        Kuhn length (nm), positive.
    temperature_k : float
        Temperature (K), positive.
    attempt_frequency_hz : float
        Attempt frequency shared by scission and re-formation (1/s), positive.
    n_min : int
        Smallest segment count considered, at least 10.
    n_max : int
        Largest segment count considered, at least ``n_min``.

    Returns
    -------
    force : float
        Mean tension at the final scission in pN, as a native Python float.

    Raises
    ------
    ValueError
        If the pulling speed, Kuhn length, temperature or attempt frequency is
        not positive, or under the conditions raised by the steps it calls.
    """
    return force
```
