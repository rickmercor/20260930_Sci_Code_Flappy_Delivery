# Adaptive orthogonal confinement for a path collective variable

## Background

A path collective variable reduces a transition between two known states to two numbers. A
progress coordinate says how far along an ordered set of reference configurations a given
configuration sits, and an orthogonal coordinate says how far off that set it lies. The
free-energy profile along the progress coordinate is only defined where sampling populates
the neighbourhood of the path, so calculations of this kind confine the orthogonal
coordinate inside a tube around the reference path. The confinement is not a numerical
convenience: it is what defines the thermodynamic ensemble whose profile is being reported,
and it is also what keeps the mapping onto the progress coordinate well conditioned and
keeps a trajectory from leaking into a neighbouring channel that shares the same range of
the progress coordinate.

A tube of prescribed width is the common choice and it has a structural defect. The
transverse stiffness of a realistic landscape varies along the route: open sites tolerate
broad transverse excursions while pinched saddles tolerate almost none. A width set too
narrow truncates the natural transverse fluctuations and writes a spurious entropic term
into the profile; a width set too wide leaves the mapping ill conditioned and lets the
trajectory cross into another channel. Because the right width is not constant, a single
number cannot serve, and the error it makes is not a constant shift but a distortion that
varies along the path and therefore moves barriers and basins relative to each other.

Two further features of this family of coordinates matter for anything quantitative. The
orthogonal coordinate of the arithmetic construction is built from a soft minimum over the
reference set, so its value on the path is not zero but a floor set by the spacing of that
set. And the arithmetic coordinate is not a physical radial displacement: for a metric that
is itself a squared displacement the coordinate grows quadratically with the transverse
excursion, so a free-energy profile read off a histogram in that coordinate carries a
coordinate-volume contribution that is not part of the physical transverse free energy.

The model system here is a cation hop between interstitial sites in an oxide layer, reduced
to a migration coordinate and the amplitude of the transverse lattice distortion that
accompanies it. The transverse stiffness rises sharply at the two saddles and the central
site carries a low transverse ridge, so the landscape has both failure modes of a fixed
width tube at once: a width that confines the saddles over-confines the sites, and a width
that fits the sites does not resolve the two sub-channels at the centre.

## Problem

A cation migrating between two interstitial sites of a model oxide layer is described by
two collective variables: a reduced migration coordinate and the amplitude, in Angstrom,
of the transverse lattice distortion that accompanies the hop. Free energies along the
migration route are computed with a path collective variable built on a fixed reference
path, and the orthogonal coordinate has to be confined so that sampling stays in a tube
around that path. A tube of fixed width is the usual choice and it is the wrong one here:
the natural transverse width of this landscape changes by an order of magnitude between
the open sites and the pinched saddles, so any single width either truncates the open
regions or fails to confine the tight ones.

Your task is to build the confinement tube whose half-width follows a fixed contour of
the orthogonal free energy instead of a prescribed distance, on the landscape below, and
to report the tube it produces.

## The landscape

Write configurations in the frame of the guide curve. A configuration is fixed by a
curve parameter t and a signed offset n along the unit normal to the curve,

    r(t) = (t, 0.15 sin(pi t)),                       t in [0, 1]
    X(t, n) = r(t) + n * N(t),        N(t) the unit normal, N = (-T_y, T_x),
                                      T(t) = r'(t) / |r'(t)|

and the potential of mean force of the two collective variables is, in kcal/mol,

    U(t, n) = V(t) + 0.5 * k(t) * n^2 + a(t) * n^3 + q(t) * n^4 + R(t, n)

    V(t) = 7.5 exp(-((t - 0.30)/0.10)^2) + 6.2 exp(-((t - 0.70)/0.10)^2)
           - 1.6 exp(-((t - 0.50)/0.11)^2)
    k(t) = 40 + 360 * [ exp(-((t - 0.30)/0.055)^2) + exp(-((t - 0.70)/0.055)^2) ]
    a(t) = 30 sin(2 pi t)
    q(t) = 140 * [ 1 + 0.6 cos(2 pi t) ]
    R(t, n) = 0.35 exp(-((t - 0.50)/0.10)^2) exp(-(n / 0.055)^2)

k is in kcal/mol/Angstrom^2, a in kcal/mol/Angstrom^3, q in kcal/mol/Angstrom^4. The two
necks at t = 0.30 and t = 0.70 are the pinched saddles; the central site carries a low
transverse ridge, so around t = 0.5 the route passes the site slightly off axis on one of
two symmetric sub-channels. The temperature is 300 K and k_B = 0.0019872041 kcal/(mol K).

Because the frame is curved, the area element of the (t, n) parameterisation is
|r'(t)| * |1 - n * c(t)| dt dn, with c(t) the signed curvature of the guide curve. Use it.

## The construction

Follow the source for every definition that the source fixes. In particular the source
fixes the arithmetic form of the two path coordinates and the metric that enters them;
the baseline the confinement is measured from; the change of variables that has to be
applied to the sampled orthogonal coordinate before a free-energy contour in it means
anything, and the exponent in that change of variables; the level the contour is measured
from and the units the contour level is expressed in; the closed-form retained fraction of
a contour-following tube for a locally harmonic well; and the relation between the
unconfined and the tube-restricted profile. None of those are supplied here beyond what
the fallback setting below has to name. The collective-variable space is two-dimensional.

These are settings of this calculation rather than claims of the source, so they are
given:

* The reference set has 200 configurations on the guide curve, equally spaced in arc length
  along the curve (not uniformly in the parameter t, and not by equal chord length), the first
  at t = 0 and the last at t = 1.
* The sharpness of the soft minimum is 2.3 divided by the mean of the path metric between
  consecutive reference configurations.
* The confinement envelope is represented on 41 nodes, the points of the guide curve
  whose on-path progress coordinate falls on a uniform grid from 0 to 1 inclusive.
  Locate them by bisection on t, and pin the first and last node to t = 0 and t = 1.
* Each node carries an orthogonal quadrature grid of 2001 uniformly spaced offsets over
  -0.34 <= n <= 0.34 Angstrom, endpoints included. That window is the sampled window.
* The two sides of each slice are folded by adding, at every positive offset n, the weight of
  the configuration at -n to the weight at +n, and the folded weight is turned into a density
  in the offset coordinate with the derivative of the offset coordinate along the positive
  branch, taken by centred differences on the offset grid.
* The half-width profile is regularised by a centred moving average over 3 nodes, with
  the profile extended beyond each end by its mirror image about the end node, the end node
  included (the point one step outside an end carries the end value), so the node count is
  preserved.
* Four contour levels are used: 1, 2, 3 and 4, in the units the source states for this
  parameter.
* An arithmetic path variable has a soft-minimum core close to the path in which the
  on-path value of the conditional free energy is not numerically well defined. Read the
  reference level the contour is measured from by linear interpolation of the conditional
  at an offset of 8.0e-05 Angstrom^2, scan outward from there, and locate the crossing of the
  prescribed rise by linear interpolation between the two bracketing points of the offset grid.
* Where the prescribed rise is never reached inside the sampled window, set the half-width
  to twice the contour level in units of thermal energy, divided by d_perp (the number of
  perpendicular directions, one less than the dimensionality of the collective-variable
  space), times the mean of the offset coordinate over the entries of the slice whose offset
  coordinate is non-negative, weighted by their statistical weights. That is the harmonic limit for an orthogonal coordinate that is a squared
  displacement. On this configuration the contour is always reached, so the fallback is
  written down only so that the behaviour is defined.

## What to report

For each of the four contour levels, integrate the smoothed envelope along the progress
coordinate over the 41 nodes with the trapezoidal rule, the abscissae being the uniform
progress-grid values 0, 0.025, ..., 1 that define the nodes (the pinned end nodes at 0 and 1).
Report the sum of those four integrals, in Angstrom^2, to six significant figures, and report
the four individual integrals as well.

Alongside the reported value, state and justify the choices that carry it: the metric and
the indexing inside the two path coordinates, the baseline the confinement is measured
against, the effective perpendicular dimensionality of this problem and where it comes
from, the exponent of the change of variables, the level the contour is measured from,
and the units of the contour parameter. Report the sharpness of the soft minimum that
follows from the reference set, and the mean on-path value of the orthogonal coordinate
over the 41 nodes. Report as well, for each contour level, the
smallest and largest envelope values, the mean retained fraction of the orthogonal
partition function inside the smoothed tube, the closed-form retained fraction for a locally harmonic well, and
the mean ratio of the unsmoothed half-width to its harmonic estimate 2 Delta F*/k(t), the
half-width of a harmonic well of the landscape's transverse stiffness k(t) at the node; say what the comparison between the mean retained fraction and its closed form,
set beside the variation of the envelope along the path, shows about the claim the source
makes for this construction; and say what the change of that ratio with the contour level
shows about the harmonic picture behind the closed form.

The quantities listed above are the reported result, not intermediate bulk output. Give them
as a compact table or a short list of labelled values inside the reasoning section, and state
each convention in a sentence or two. A short table of exactly those values is not the kind of
per-iteration or per-candidate output the format note below asks you to leave out, and a
response that reports them compactly is both complete and within the length the note asks for.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

reference_path

Goal
----
Lay down the ordered set of reference configurations that defines the path collective variable, taking the guide curve of the configuration as the route: n_images points on the guide curve r(t) = (t, 0.15 sin(pi t)), equally spaced in arc length along the curve (not uniformly in the parameter t, and not by equal chord length), with the first at t = 0 and the last at t = 1, every configuration placed to within 1e-9 Angstrom of its exact equal-arc-length position (an arc-length table of at least 40 000 points over the curve, or exact quadrature of the speed with a root finder, achieves this; a 10 000-point table does not). Return the configurations in the order in which they run from the initial site to the final site, one row per configuration.

```python
def reference_path(n_images: int) -> "np.ndarray":
    """Lay down the ordered set of reference configurations that defines the path collective variable,
    taking the guide curve of the configuration as the route: n_images points on the guide curve
    r(t) = (t, 0.15 sin(pi t)), equally spaced in arc length along the curve (not uniformly in the
    parameter t, and not by equal chord length), with the first at t = 0 and the last at t = 1,
    every configuration placed to within 1e-9 Angstrom of its exact equal-arc-length position (an
    arc-length table of at least 40 000 points over the curve, or exact quadrature of the speed with
    a root finder, achieves this; a 10 000-point table does not). Return the configurations in the
    order in which they run from the initial site to the final site, one row per configuration.

    Args:
        n_images: int, the number of reference configurations (at least 2).

    Returns:
        An (n_images, 2) float64 array of reference configurations in the collective-variable plane,
        first column the migration coordinate and second column the transverse distortion in
        Angstrom.

    Raises:
        ValueError: if n_images is smaller than 2.
    """
    return None
```

### Step 2

pcv_coordinates

Goal
----
Evaluate the progress coordinate and the orthogonal coordinate of the arithmetic path collective variable for a batch of configurations, exactly as the source defines the pair. Two choices inside that definition are load-bearing and the source states both: which quantity plays the role of the inter-configuration metric, and how the reference configurations are indexed in the numerator of the progress coordinate. Set the sharpness of the soft minimum from the reference set itself, as 2.3 divided by the mean of that metric between consecutive reference configurations. Guard the exponentials against underflow; the result must not depend on how that guarding is done.

```python
def pcv_coordinates(points: "np.ndarray", images: "np.ndarray") -> "np.ndarray":
    """Evaluate the progress coordinate and the orthogonal coordinate of the arithmetic path collective
    variable for a batch of configurations, exactly as the source defines the pair. Two choices
    inside that definition are load-bearing and the source states both: which quantity plays the
    role of the inter-configuration metric, and how the reference configurations are indexed in the
    numerator of the progress coordinate. Set the sharpness of the soft minimum from the reference
    set itself, as 2.3 divided by the mean of that metric between consecutive reference
    configurations. Guard the exponentials against underflow; the result must not depend on how that
    guarding is done.

    Args:
        points: array-like of shape (n_points, 2) (a single (2,) point is accepted), configurations
            in the collective-variable plane: migration coordinate, transverse distortion in
            Angstrom.
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.

    Returns:
        An (n_points, 2) float64 array whose first column is the progress coordinate and whose
        second column is the orthogonal coordinate.

    Raises:
        ValueError: if points or images do not have two columns, or if fewer than two reference
        configurations are given.
    """
    return None
```

### Step 3

station_frame

Goal
----
Place the nodes on which the confinement envelope will be represented. The source represents the envelope on a grid in the progress coordinate, so the nodes are the points of the guide curve whose on-path progress coordinate falls on a uniform grid spanning the whole path, endpoints included. Locate each node to machine precision by bisection on the guide-curve parameter rather than by interpolating a table, and pin the two end nodes to the two ends of the curve. For each node also report the on-path value of the orthogonal coordinate, which the source needs as a baseline because that value is not zero, together with the speed and the signed curvature of the guide curve there.

```python
def station_frame(images: "np.ndarray", n_stations: int) -> "np.ndarray":
    """Place the nodes on which the confinement envelope will be represented. The source represents the
    envelope on a grid in the progress coordinate, so the nodes are the points of the guide curve
    whose on-path progress coordinate falls on a uniform grid spanning the whole path, endpoints
    included. Locate each node to machine precision by bisection on the guide-curve parameter rather
    than by interpolating a table, and pin the two end nodes to the two ends of the curve. For each
    node also report the on-path value of the orthogonal coordinate, which the source needs as a
    baseline because that value is not zero, together with the speed and the signed curvature of the
    guide curve there.

    Args:
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.
        n_stations: int, the number of envelope nodes (at least 3), uniform in the progress
            coordinate from 0 to 1 inclusive.

    Returns:
        An (n_stations, 5) float64 array with columns [progress coordinate, guide-curve parameter,
        on-path orthogonal coordinate, guide-curve speed, signed guide-curve curvature].

    Raises:
        ValueError: if n_stations is smaller than 3, or if fewer than two reference configurations
        are given.
    """
    return None
```

### Step 4

orthogonal_slices

Goal
----
Build the orthogonal slice of the ensemble at every node. Step off the guide curve along the unit normal on a uniform grid of signed offsets covering the sampling window symmetrically, endpoints included, map each offset to a configuration in the collective-variable plane, and report two numbers for it: the offset coordinate that the confinement acts on, which the source defines as the orthogonal coordinate measured from the on-path baseline of that node rather than from zero; and the equilibrium statistical weight of the configuration. The weight is the Boltzmann factor of the landscape at 300 K together with the volume element of the curved frame; shift the exponent by the global minimum of the landscape over the grid so the weights stay in range, which changes every weight by one common factor and cancels everywhere it is used.

```python
def orthogonal_slices(stations: "np.ndarray", images: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    """Build the orthogonal slice of the ensemble at every node. Step off the guide curve along the
    unit normal on a uniform grid of signed offsets covering the sampling window symmetrically,
    endpoints included, map each offset to a configuration in the collective-variable plane, and
    report two numbers for it: the offset coordinate that the confinement acts on, which the source
    defines as the orthogonal coordinate measured from the on-path baseline of that node rather than
    from zero; and the equilibrium statistical weight of the configuration. The weight is the
    Boltzmann factor of the landscape at 300 K together with the volume element of the curved frame;
    shift the exponent by the global minimum of the landscape over the grid so the weights stay in
    range, which changes every weight by one common factor and cancels everywhere it is used.

    Args:
        stations: array-like of shape (n_stations, 5), the node table returned by station_frame.
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).

    Returns:
        A (n_stations, n_ortho, 2) float64 array whose last axis holds [offset coordinate,
        statistical weight].

    Raises:
        ValueError: if stations does not have five columns, if n_ortho is even or smaller than 3, or
        if n_max is not positive.
    """
    return None
```

### Step 5

conditional_free_energy

Goal
----
Turn each orthogonal slice into the conditional free energy of the offset coordinate that the contour condition is applied to. Three things have to happen and the source fixes all three. The offset coordinate cannot tell the two sides of the path apart, so the two branches of the slice have to be brought together before anything is measured: at every positive normal offset add the statistical weight of the configuration at the opposite offset (minus n) to the weight at plus n, and attach the sum to the offset coordinate of the positive branch. The density is wanted in the offset coordinate while the slice is sampled in the normal offset, so the change of variables between them has to be carried through: divide the folded weight by the magnitude of the derivative of the offset coordinate with respect to the normal offset along the positive branch, taken by centred finite differences on the offset grid. And a histogram in the offset coordinate carries a coordinate-volume contribution that the source removes with an explicit metric correction whose exponent is fixed by the effective perpendicular dimensionality of the problem, itself fixed by the dimensionality of the collective-variable space. Return the positive branch sorted by increasing offset, with the free energy on an absolute scale in kcal/mol.

```python
def conditional_free_energy(slices: "np.ndarray", n_ortho: int, n_max: float, d_eff: int) -> "np.ndarray":
    """Turn each orthogonal slice into the conditional free energy of the offset coordinate that the
    contour condition is applied to. Three things have to happen and the source fixes all three. The
    offset coordinate cannot tell the two sides of the path apart, so the two branches of the slice
    have to be brought together before anything is measured: at every positive normal offset add the
    statistical weight of the configuration at the opposite offset (minus n) to the weight at plus
    n, and attach the sum to the offset coordinate of the positive branch. The density is wanted in
    the offset coordinate while the slice is sampled in the normal offset, so the change of
    variables between them has to be carried through: divide the folded weight by the magnitude of
    the derivative of the offset coordinate with respect to the normal offset along the positive
    branch, taken by centred finite differences on the offset grid. And a histogram in the offset
    coordinate carries a coordinate-volume contribution that the source removes with an explicit
    metric correction whose exponent is fixed by the effective perpendicular dimensionality of the
    problem, itself fixed by the dimensionality of the collective-variable space. Return the
    positive branch sorted by increasing offset, with the free energy on an absolute scale in
    kcal/mol.

    Args:
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.

    Returns:
        A (n_stations, (n_ortho - 1)//2, 2) float64 array whose last axis holds [offset coordinate,
        conditional free energy in kcal/mol], sorted by increasing offset within each station.

    Raises:
        ValueError: if slices does not have shape (n_stations, n_ortho, 2), or if d_eff is smaller
        than 2.
    """
    return None
```

### Step 6

adaptive_half_width

Goal
----
Find the half-width of the adaptive tube at every node: the offset at which the conditional free energy first rises by the prescribed amount above its on-path reference level, which is the level the source measures the excess from. An arithmetic path variable has a soft-minimum core in which the on-path value is not numerically well defined, so read the reference level by linear interpolation of the conditional at the prescribed reference offset and scan outward from there; interpolate the crossing itself linearly between the two bracketing nodes of the offset grid. Where the prescribed rise is never reached inside the sampled window, set the half-width to twice the contour level in units of thermal energy, divided by d_perp (the number of perpendicular directions, one less than the dimensionality of the collective-variable space), times the mean of the offset coordinate over the entries of the slice whose offset coordinate is non-negative, weighted by their statistical weights; that is the harmonic limit for an orthogonal coordinate that is a squared displacement. Return one half-width per node, in the units of the offset coordinate.

```python
def adaptive_half_width(cond: "np.ndarray", slices: "np.ndarray", delta_f_star: float, d_eff: int, dz_ref: float) -> "np.ndarray":
    """Find the half-width of the adaptive tube at every node: the offset at which the conditional free
    energy first rises by the prescribed amount above its on-path reference level, which is the
    level the source measures the excess from. An arithmetic path variable has a soft-minimum core
    in which the on-path value is not numerically well defined, so read the reference level by
    linear interpolation of the conditional at the prescribed reference offset and scan outward from
    there; interpolate the crossing itself linearly between the two bracketing nodes of the offset
    grid. Where the prescribed rise is never reached inside the sampled window, set the half-width
    to twice the contour level in units of thermal energy, divided by d_perp (the number of
    perpendicular directions, one less than the dimensionality of the collective-variable space),
    times the mean of the offset coordinate over the entries of the slice whose offset coordinate is
    non-negative, weighted by their statistical weights; that is the harmonic limit for an
    orthogonal coordinate that is a squared displacement. Return one half-width per node, in the
    units of the offset coordinate.

    Args:
        cond: array-like of shape (n_stations, (n_ortho - 1)//2, 2), the conditional free energies
            returned by conditional_free_energy.
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        delta_f_star: float, the positive contour level in kcal/mol (the level in units of k_B T
            times k_B T).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        dz_ref: float, the positive offset in Angstrom^2 at which the reference level of the
            conditional free energy is read.

    Returns:
        A (n_stations,) float64 array of tube half-widths in the offset coordinate.

    Raises:
        ValueError: if cond does not have shape (n_stations, n_half, 2), if delta_f_star or dz_ref
        is not positive, or if d_eff is smaller than 2.
    """
    return None
```

### Step 7

smooth_envelope

Goal
----
Regularise the half-width profile along the progress coordinate into the continuous envelope the restraint reads, by a centred moving average over an odd number of nodes. Replace any node that carries no value by the mean of the nodes that do before averaging; extend the profile beyond each end by its mirror image about the end node with the end node included, so that the point one step outside an end carries the end value itself (numpy's symmetric padding, not the reflect padding that omits the end value), and the averaged profile keeps the same number of nodes; return the profile unchanged when the window is a single node.

```python
def smooth_envelope(profile: "np.ndarray", window: int) -> "np.ndarray":
    """Regularise the half-width profile along the progress coordinate into the continuous envelope the
    restraint reads, by a centred moving average over an odd number of nodes. Replace any node that
    carries no value by the mean of the nodes that do before averaging; extend the profile beyond
    each end by its mirror image about the end node with the end node included, so that the point
    one step outside an end carries the end value itself (numpy's symmetric padding, not the reflect
    padding that omits the end value), and the averaged profile keeps the same number of nodes;
    return the profile unchanged when the window is a single node.

    Args:
        profile: array-like of shape (n_stations,), the half-width profile; entries may be NaN where
            no value was found.
        window: int, a positive odd number of nodes for the centred moving average.

    Returns:
        A (n_stations,) float64 array, the smoothed envelope.

    Raises:
        ValueError: if window is not a positive odd integer.
    """
    return None
```

### Step 8

tube_projected_profile

Goal
----
Report, at every node, the unconfined free-energy profile along the path, the fraction of the orthogonal partition function that the given tube retains, and the profile that a calculation inside that tube would return. The first is the marginal of the equilibrium density over the sampled orthogonal window, with the same grid spacing used as the measure. The third follows from the first two by the exact relation the source derives between them, which is a two-line consequence of the definitions of the marginal and of the restricted marginal; get its sign right.

```python
def tube_projected_profile(slices: "np.ndarray", z_max: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    """Report, at every node, the unconfined free-energy profile along the path, the fraction of the
    orthogonal partition function that the given tube retains, and the profile that a calculation
    inside that tube would return. The first is the marginal of the equilibrium density over the
    sampled orthogonal window, with the same grid spacing used as the measure. The third follows
    from the first two by the exact relation the source derives between them, which is a two-line
    consequence of the definitions of the marginal and of the restricted marginal; get its sign
    right.

    Args:
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        z_max: array-like of shape (n_stations,), the tube half-width at every node, in the offset
            coordinate.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).

    Returns:
        A (n_stations, 3) float64 array with columns [unconfined free energy in kcal/mol, retained
        fraction, tube-restricted free energy in kcal/mol].

    Raises:
        ValueError: if z_max does not carry one entry per station.
    """
    return None
```

### Step 9

harmonic_tube_reference

Goal
----
Write down the closed-form predictions the source derives for a locally harmonic orthogonal well, for a given contour level and an array of local perpendicular stiffnesses: the fraction of the orthogonal partition function that a contour-following tube retains, which the source obtains in closed form and which depends on the stiffness through nothing at all; the physical half-width of that tube; and the position of the wall expressed in local thermal widths. Return one row per stiffness. Do not compute these numerically; they are the analytic limits the audit is checked against.

```python
def harmonic_tube_reference(d_eff: int, delta_f_star: float, stiffness: "np.ndarray") -> "np.ndarray":
    """Write down the closed-form predictions the source derives for a locally harmonic orthogonal
    well, for a given contour level and an array of local perpendicular stiffnesses: the fraction of
    the orthogonal partition function that a contour-following tube retains, which the source
    obtains in closed form and which depends on the stiffness through nothing at all; the physical
    half-width of that tube; and the position of the wall expressed in local thermal widths. Return
    one row per stiffness. Do not compute these numerically; they are the analytic limits the audit
    is checked against.

    Args:
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        delta_f_star: float, the positive contour level in kcal/mol (the level in units of k_B T
            times k_B T).
        stiffness: array-like of shape (n_stiffness,) (a scalar is accepted), positive local
            perpendicular stiffnesses in kcal/mol/Angstrom^2.

    Returns:
        A (n_stiffness, 3) float64 array with columns [retained fraction, physical tube half-width
        in Angstrom, wall position in local thermal widths].

    Raises:
        ValueError: if d_eff is smaller than 2, or if delta_f_star or any stiffness is not positive.
    """
    return None
```

### Step 10

tube_audit

Goal
----
Run the whole construction on the configuration and report it. Build the reference set, place the nodes, take the orthogonal slices, and then for each contour level in turn build the conditional free energies, find the adaptive half-widths, smooth them into the envelope, project the profile through the tube, and evaluate the closed-form predictions at the local stiffnesses of the nodes, the transverse stiffness k(t) of the landscape at each node's curve parameter. Record one row per contour level, and put the run-level quantities in a leading row. The integral of the envelope along the progress coordinate is taken with the trapezoidal rule over the node grid, the abscissae being the uniform progress-grid values from 0 to 1 inclusive that define the nodes, and the reported total is the sum of those integrals over the contour levels.

```python
def tube_audit(n_images: int, n_stations: int, n_ortho: int, n_max: float, d_eff: int, smooth_window: int, dz_ref: float, contours: tuple) -> "np.ndarray":
    """Run the whole construction on the configuration and report it. Build the reference set, place
    the nodes, take the orthogonal slices, and then for each contour level in turn build the
    conditional free energies, find the adaptive half-widths, smooth them into the envelope, project
    the profile through the tube, and evaluate the closed-form predictions at the local stiffnesses
    of the nodes, the transverse stiffness k(t) of the landscape at each node's curve parameter.
    Record one row per contour level, and put the run-level quantities in a leading row. The
    integral of the envelope along the progress coordinate is taken with the trapezoidal rule over
    the node grid, the abscissae being the uniform progress-grid values from 0 to 1 inclusive that
    define the nodes, and the reported total is the sum of those integrals over the contour levels.

    Args:
        n_images: int, the number of reference configurations (at least 2).
        n_stations: int, the number of envelope nodes (at least 3), uniform in the progress
            coordinate from 0 to 1 inclusive.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        smooth_window: int, a positive odd number of nodes for the moving average of the half-width
            profile.
        dz_ref: float, the positive offset in Angstrom^2 at which the reference level of the
            conditional free energy is read.
        contours: sequence of positive floats, the contour levels in units of k_B T (at least one).

    Returns:
        A (len(contours) + 1, 9) float64 array. Row zero holds [reported total, soft-minimum
        sharpness, mean on-path orthogonal baseline at the nodes, smallest node stiffness, largest
        node stiffness, mean orthogonal coordinate of the reference configurations themselves, span
        of the progress coordinate over the reference configurations, 0, 0]. Each later row holds
        [contour level, contour level in kcal/mol, integral of the envelope along the progress
        coordinate, smallest envelope value, largest envelope value, mean retained fraction, closed-
        form retained fraction, mean ratio of the unsmoothed half-width to its harmonic estimate 2
        Delta F*/k(t) (the half-width of a harmonic well of stiffness k(t) at the same excess, in
        the offset coordinate), span of the tube-restricted profile in kcal/mol].

    Raises:
        ValueError: if contours is empty, or for any invalid argument of the underlying steps
        (n_images below 2, n_stations below 3, an even n_ortho, a non-positive n_max or dz_ref,
        d_eff below 2, an even smooth_window).
    """
    return None
```
