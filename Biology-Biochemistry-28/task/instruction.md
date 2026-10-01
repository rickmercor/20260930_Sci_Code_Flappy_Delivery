# Biology-Biochemistry-28

## Background

Epithelial monolayers are confluent sheets of polygonal cells whose shapes are set by cell-area elasticity and junctional tension, and whose junctions remodel through T1 neighbour exchanges. Under sustained stretching such sheets, like ductile solids, can localise deformation into a propagating neck, and vertex models connect that tissue-scale instability to the shape changes and rearrangements of individual cells.

## Problem

Stretched epithelial sheets can neck: after a stage of uniform elongation a narrowed region forms and then sweeps along the tissue at a nearly constant load. I model an ordered sheet with a vertex model in which each cell has the dimensionless energy e = (a − 1)²/2 + κ(p − χ)²/2 of its area a and perimeter p (lengths in units of the square root of the preferred cell area), with κ = 0.35 and χ = 2.45. The stripe is the regular honeycomb relaxed to its stress-free state, pulled along a lattice direction perpendicular to one pair of edges of every cell with its lateral boundaries free; the stretch λ is measured from the stress-free honeycomb, and stress means nominal stress referred to that configuration. Each region of the stripe deforms homogeneously: every cell's extents along and across the load follow the tissue stretches while the rest of its shape relaxes to minimum energy. A load-perpendicular edge undergoes a T1 neighbour exchange once its length has shrunk to 0.08; in the mean-field description I use, the tissue stretch is continuous through the exchange, each four-cell unit around such an edge keeps its length along the load while the two cells that shared the edge separate and the two cells at its ends come into contact, and the stripe afterwards deforms homogeneously as the same honeycomb turned by 90°. From this description I want the steady necking-propagation nominal stress divided by the nominal stress at necking bifurcation, reported to ten significant figures.

In the reasoning, state the necking-bifurcation stretch and its nominal stress; the cellular stretch along the load just after the exchange; the propagation nominal stress together with the stretches of the un-necked and necked regions; how the post-exchange branch of the stress–stretch relation you use relates to the published mean-field relation for this model; and the final ratio. These are the quantities that determine the final number; there is no need to restate the supplied model.

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

01_solve_stress_free_scale

Goal
----
Find the edge-length scale of the regular hexagonal cell that is stress-free under the area-perimeter cell energy.

```python
def solve_stress_free_scale(kappa: float, chi: float) -> float:
    """Return the edge-length scale of the stress-free regular hexagonal cell.

    A cell of dimensionless area ``a`` and perimeter ``p`` carries the energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2``. Among regular
    hexagons, return the ratio ``k`` of the edge length that minimises ``e``
    to the edge length of the regular hexagon of unit area, with relative
    accuracy ``1e-13``.

    Parameters
    ----------
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter (shape index).

    Returns
    -------
    float
        The edge-length scale ``k``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number
        (booleans are rejected).
    """
    return 0.0
```

### Step 2

02_compute_honeycomb_stresses

Goal
----
Evaluate the two nominal stresses of an ordered honeycomb whose cells follow a homogeneous tissue stretch while their remaining vertex freedom relaxes.

```python
def compute_honeycomb_stresses(
    stretch: float,
    lateral_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Return the two nominal stresses of a homogeneously stretched honeycomb.

    The reference tissue is the honeycomb of regular hexagons whose edge
    length is ``edge_scale`` times that of the unit-area regular hexagon,
    oriented so that one pair of edges of every cell is perpendicular to the
    load. The tissue is stretched by ``stretch`` along the load and by
    ``lateral_stretch`` across it: every cell's extent along the load (the
    lattice period in that direction) and across it (the spacing of the
    cell rows) scale with these stretches, and the remaining vertex degree
    of freedom of each cell relaxes to minimise the cell energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2`` of its dimensionless
    area ``a`` and perimeter ``p``.

    Return the nominal stresses conjugate to the two stretches: the partial
    derivatives of the elastic energy per unit reference area with respect
    to ``stretch`` and to ``lateral_stretch``.

    Parameters
    ----------
    stretch : float
        Positive stretch along the load.
    lateral_stretch : float
        Positive stretch across the load.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    np.ndarray
        Float array ``[along, across]`` of the two nominal stresses.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), or if the smallest perimeter a cell with these extents can
        have is below ``chi`` by more than a relative ``1e-12`` (edges not
        under tension, a regime this model does not cover), or if the relaxed
        120-degree shape would have a negative load-perpendicular edge by more
        than a relative ``1e-12``.
    """
    return stresses
```

### Step 3

03_solve_uniaxial_state

Goal
----
Find the lateral stretch and the along-load nominal stress of the honeycomb stripe pulled along the load with free lateral boundaries.

```python
def solve_uniaxial_state(
    stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Return the along-load nominal stress and lateral stretch under free lateral boundaries.

    The honeycomb of ``compute_honeycomb_stresses`` is stretched by
    ``stretch`` along the load while its lateral boundaries carry no load:
    the lateral stretch is the positive value at which the across-load
    nominal stress vanishes. Return that lateral stretch and the along-load
    nominal stress of the resulting state, both with absolute accuracy
    ``1e-12``.

    Parameters
    ----------
    stretch : float
        Positive stretch along the load.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    np.ndarray
        Float array ``[stress, lateral_stretch]``.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if no positive lateral stretch makes the across-load
        nominal stress vanish, or if ``compute_honeycomb_stresses`` rejects
        the resulting state.
    """
    return state
```

### Step 4

04_locate_rearrangement_stretch

Goal
----
Find the stretch at which the load-perpendicular cell edges of the freely contracting stripe shrink to the rearrangement threshold.

```python
def locate_rearrangement_stretch(
    threshold: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the stretch at which the load-perpendicular edges reach the rearrangement threshold.

    Under the free-lateral stretching of ``solve_uniaxial_state``, the cell
    edges perpendicular to the load shorten as the stretch grows. Return the
    smallest stretch in ``(1, 10]`` at which their length equals
    ``threshold`` (in the units of the cell energy, the square root of the
    preferred cell area), with absolute accuracy ``1e-12``. Beyond that
    first crossing the edge keeps shrinking and can turn negative, where
    ``solve_uniaxial_state`` rejects the stretch, so locate the crossing
    from below (for example by stepping up from stretch 1) rather than by
    evaluating the edge at stretch 10.


    Parameters
    ----------
    threshold : float
        Positive edge length at which the edge rearranges.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    float
        The rearrangement stretch.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if the load-perpendicular edge at stretch 1 is not longer
        than ``threshold``, if that edge does not shrink to ``threshold`` for
        any stretch up to 10, or if ``solve_uniaxial_state`` rejects a
        stretch before the edge reaches ``threshold``.
    """
    return 0.0
```

### Step 5

05_compute_nominal_stress

Goal
----
Evaluate the homogeneous-deformation nominal stress of the stripe on both sides of the collective T1 rearrangement.

```python
def compute_nominal_stress(
    stretch: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the nominal stress of the stripe across the rearrangement.

    For ``stretch <= transition_stretch`` the stripe is the honeycomb of
    ``solve_uniaxial_state`` stretched by ``stretch``. At
    ``transition_stretch`` every four-cell unit built around a
    load-perpendicular edge exchanges neighbours: the unit keeps its length
    along the load but now holds three cells along the load where it held
    two, and the stripe becomes the same honeycomb turned by 90 degrees (one
    pair of edges of every cell parallel to the load). For
    ``stretch > transition_stretch`` the turned honeycomb is stretched along
    the load in proportion to the tissue stretch, with free lateral
    boundaries. Return the nominal stress, the derivative with respect to
    ``stretch`` of the elastic energy per unit reference area, with absolute
    accuracy ``1e-12``.

    Parameters
    ----------
    stretch : float
        Positive tissue stretch along the load.
    transition_stretch : float
        Stretch of the rearrangement; must exceed 1.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    float
        The nominal stress along the load.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``transition_stretch`` does not exceed 1, or if
        ``solve_uniaxial_state`` rejects the cell state reached.
    """
    return 0.0
```

### Step 6

06_compute_stretching_work

Goal
----
Integrate the stripe's nominal stress over a stretch interval that may straddle the rearrangement.

```python
def compute_stretching_work(
    stretch_start: float,
    stretch_end: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the work per unit reference area between two stretches.

    Return the integral of ``compute_nominal_stress`` (with the given
    ``transition_stretch``, ``edge_scale``, ``kappa`` and ``chi``) with
    respect to the stretch from ``stretch_start`` to ``stretch_end``, with
    absolute accuracy ``1e-12``. The integral is negative when
    ``stretch_end < stretch_start``.

    Parameters
    ----------
    stretch_start : float
        Positive lower limit of integration.
    stretch_end : float
        Positive upper limit of integration.
    transition_stretch : float
        Stretch of the rearrangement; must exceed 1.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    float
        The work per unit reference area.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``transition_stretch`` does not exceed 1, or if
        ``solve_uniaxial_state`` rejects a cell state on the integration path.
    """
    return 0.0
```

### Step 7

07_solve_propagation_state

Goal
----
Find the nominal stress and the two coexisting stretches at which a neck propagates steadily along the stripe.

```python
def solve_propagation_state(
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
    tolerance: float,
) -> "np.ndarray":
    """Return the steady necking-propagation stress and the coexisting stretches.

    With the constitutive law of ``compute_nominal_stress``, find the
    nominal stress ``s`` at which an un-necked region at stretch ``u`` in
    ``[1, transition_stretch]`` and a necked region at stretch
    ``m > transition_stretch`` coexist in steady propagation: both carry
    ``s``, and the work ``s * (m - u)`` done on unit reference area passing
    from ``u`` to ``m`` equals ``compute_stretching_work`` from ``u`` to
    ``m``. The admissible ``s`` lies between the larger of the stresses at
    stretch 1 and immediately above ``transition_stretch``, and the stress at
    ``transition_stretch``. The upper end is the smaller of that transition
    stress and the largest stress reached before the connected admissible
    necked branch ends; over this range the stress rises with the stretch on
    each branch. Return ``s`` with absolute accuracy ``tolerance`` and the
    stretches ``u`` and ``m`` carrying ``s`` on the two branches.

    Parameters
    ----------
    transition_stretch : float
        Stretch of the rearrangement; must exceed 1.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    tolerance : float
        Positive absolute accuracy of ``s``, at most ``1e-6``.

    Returns
    -------
    np.ndarray
        Float array ``[s, u, m]``.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``tolerance`` exceeds ``1e-6``, if the stress
        immediately above ``transition_stretch`` or at stretch 1 is not below
        the stress at ``transition_stretch``, if no admissible ``s`` balances
        the work, or if an earlier step rejects a state it visits.
    """
    return state
```

### Step 8

08_estimate_propagation_ratio

Goal
----
Compose every earlier step to obtain the ratio of the steady necking-propagation stress to the necking-bifurcation stress of the ordered stripe.

```python
def estimate_propagation_ratio(
    kappa: float = 0.35,
    chi: float = 2.45,
    threshold: float = 0.08,
    tolerance: float = 1e-13,
) -> float:
    """Return the ratio of the propagation stress to the bifurcation stress.

    The reference honeycomb uses the edge-length scale of
    ``solve_stress_free_scale``. Its load-perpendicular edges rearrange at the
    stretch returned by ``locate_rearrangement_stretch`` for ``threshold``;
    the bifurcation stress is ``compute_nominal_stress`` at that stretch; the
    propagation stress is the first entry of ``solve_propagation_state`` for
    that stretch and ``tolerance``. Return the propagation stress divided by
    the bifurcation stress. The defaults reproduce the problem statement.

    Parameters
    ----------
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    threshold : float
        Positive edge length at which load-perpendicular edges rearrange.
    tolerance : float
        Positive absolute accuracy of the propagation stress, at most 1e-6.

    Returns
    -------
    float
        The propagation-to-bifurcation stress ratio.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if the nominal stress is not at a load maximum at the
        rearrangement stretch (still rising at ``(1 - 1e-6)`` times that
        stretch and lower immediately above it), or if any earlier step
        rejects its input.
    """
    return 0.0
```
