# Physics-Computational_Physics-3

## Problem

For the published adaptive surface-particle model, determine the occupied-area-weighted first-coordinate second moment after the finite local inspection experiment below; the model's published mathematical definitions govern where its public implementation differs, and surface geometry is analytic.

The surface is \(x^2+(y/0.8)^2+(z/0.6)^2=1\), and initial labels \(i=0,\ldots,47\) have \(t_i=1-2(i+0.5)/48\), \(\phi_i=i\pi(3-\sqrt5)\), and positions \((\sqrt{1-t_i^2}\cos\phi_i,0.8\sqrt{1-t_i^2}\sin\phi_i,0.6t_i)\); delete those with \(x>0.6\) and \(z>0.1\), retaining labels. Let \(c\) be the surviving position with label 34, \(n\) its outward unit normal, \(e=(n\times(0,0,1))/\|n\times(0,0,1)\|\), and \(f=n\times e\), and include five further initial particles with labels \(100+k\), \(k=0,\ldots,4\), at the Euclidean closest surface points to \(c+0.12[\cos(2\pi k/5)e+\sin(2\pi k/5)f]\). Use reference spacing \(0.42\), curvature weight \(0.5\), reference curvature zero, physical length-field neighborhood radius \(0.55\), support-area coefficient exactly \(0.933\), lower and upper support thresholds \(0.7\) and \(1.25\), and the componentwise insertion perturbation \((0.13,-0.21,0.07)\); every kernel sum centered on a particle uses that particle's own characteristic length in the kernel.

The local inspection queue is exactly \((26,34,26)\), with the model's particle-number rule applied once at each visited label using the cloud present at that visit; geometric relaxation is absent, and the experiment ends after these three visits. Return \(M=\sum_i A_i x_i^2/\sum_i A_i\), where \(A_i\) is the model's occupied-area estimate on the delivered final cloud and \(x_i\) is the first Cartesian coordinate, to absolute tolerance \(10^{-7}\) in squared-length units; no additional random draws or stopping decisions are part of the instance.

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

01_surface_geometry

Goal
----
Analytic geometry of a smooth ellipsoid at surface sample positions.

The surface is sum_d (x_d/a_d)^2=1. Geometry is supplied analytically in this
task rather than estimated from a background point cloud. The scalar curvature
measure reported in column 3 is the Euclidean norm of the two principal
curvatures at the sample.

```python
def surface_geometry(points: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Return an (n,4) float array in the input row order.

    Columns 0 to 2 hold the outward unit normal at each sample; column 3 holds
    the scalar curvature measure described above. points has shape (n,3) with
    n>=1 and axes has shape (3,) with every semiaxis strictly positive. All
    entries are finite and every sample satisfies the surface equation to
    absolute residual 1e-8 or less.

    Raises
    ------
    ValueError
        If points or axes has the wrong shape, if n<1, if any entry of points
        or axes is not finite, if any semiaxis is not strictly positive, or if
        any sample exceeds the on-surface residual bound.
    """
    return
```

### Step 2

02_characteristic_lengths

Goal
----
Curvature-dependent target spacing with a finite physical neighborhood.

curvature supplies the scalar curvature measure of the previous step at each sample. kref is a reference value carried in the same inverse-length units as curvature, and tau weights the departure of the local curvature from it. radius is a physical Euclidean distance in the same length units as points, never a multiple of a spacing. Column 0 is the spacing this task attributes to a sample from its own curvature alone; column 1 is the spacing it attributes to the same sample once the samples lying within radius of it are taken into account.

Definitions used by this task (all arrays indexed in input row order, and d_ij the Euclidean distance between samples i and j):

raw_i = h0 / sqrt(1 + tau * |curvature_i - kref|) h_i = min over all j with d_ij <= radius of raw_j (j = i included)

Column 0 is raw_i and column 1 is h_i. The neighborhood includes its center and its boundary.

```python
def characteristic_lengths(points: 'np.ndarray | list | tuple', curvature: 'np.ndarray | list | tuple', h0: float, tau: float, kref: float, radius: float) -> 'np.ndarray':
    """Return an (n,2) float array in the input row order.

    Columns are [own-curvature spacing, characteristic spacing], both strictly
    positive lengths in the units of points. points has shape (n,3) with n>=1
    and curvature has shape (n,) with every entry nonnegative. h0 and radius
    are strictly positive scalars; tau and kref are nonnegative scalars. All
    array entries and all scalars are finite.

    Raises
    ------
    ValueError
        If points or curvature has the wrong shape, if n<1, if any entry of
        points or curvature is not finite, if any curvature entry is negative,
        if h0 or radius is not strictly positive, if tau or kref is negative,
        or if any of h0, tau, kref or radius is not finite.
    """
    return
```

### Step 3

03_occupied_support

Goal
----
Occupied areas and local integral support for variable target resolution.

lengths supplies the characteristic spacing of the previous step at each sample. packing is a dimensionless positive coefficient of this task's reference-area convention. Column 0 is the area this task attributes to a sample on the surface; column 1 is the dimensionless support that measures that attribution against the sample's own target resolution. Both columns are evaluated on the positions supplied in this call, never on cached positions carried over from an earlier state.

Definitions used by this task (d_ij the Euclidean distance between samples, h_i the characteristic spacing of sample i, sums over every sample j including j = i, whose distance is 0):

r_i = 2 * h_i W_ij = 3 / (pi * r_i^2) * max(1 - d_ij / r_i, 0) rho_i = sum_j (h_j / h_i)^2 * W_ij A_i = 1 / rho_i S_i = packing * h_i^2 * rho_i

Column 0 is A_i and column 1 is S_i. Every kernel sum centered on sample i uses r_i, the cutoff built from that sample's own spacing, for every term; a neighbor at d_ij >= r_i contributes exactly zero.

```python
def occupied_support(points: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', packing: float=0.933) -> 'np.ndarray':
    """Return an (n,2) float array in the input row order.

    Columns are [occupied area A_i, dimensionless support S_i]; column 0
    carries squared-length units and column 1 is dimensionless. points has
    shape (n,3) with n>=1 and lengths has shape (n,) with every entry strictly
    positive. packing is a strictly positive finite scalar. All entries are
    finite. Two distinct samples closer than 1e-12 lie outside this contract.
    Any finite-range sum evaluated for a sample takes its range from that
    sample's own characteristic spacing, never from another sample's spacing
    or from any combination of two samples' spacings.

    Raises
    ------
    ValueError
        If points or lengths has the wrong shape, if n<1, if any entry of
        points, lengths or packing is not finite, if any entry of lengths or
        packing is not strictly positive, or if two distinct samples are
        separated by less than 1e-12.
    """
    return
```

### Step 4

04_event_proposal

Goal
----
A support-triggered local adaptation proposal before surface return.

Inspect the single particle whose identifier equals target, using the support values supplied in this call, and report which of the three admissible outcomes that particle's support selects, together with the trial position that outcome proposes. normals supplies the outward unit normal at each sample; mu supplies a three-component perturbation consumed by the trial construction. No particle is added, removed or moved here, and no trial is returned to the surface here.

Definitions used by this task. Let k be the row whose identifier equals target, S_k its supplied support, h_k its characteristic spacing, x_k its position and n_k its outward unit normal.

Merge outcome, selected when S_k > upper: the partner j is the other particle at the smallest Euclidean distance from x_k (equal distances select the smaller identifier). Return [-1, ids[j], midpoint of x_k and x_j].

Insertion outcome, selected when S_k < lower: let rho_k(x) be the weighted density defined in the preceding step, with central evaluation position x and all characteristic lengths and other particle positions fixed. Its central self contribution is constant; a neighbor exactly at the kernel cutoff has zero derivative contribution. Define the tangential gradient G = (I - n_k n_k^T) grad rho_k evaluated at x_k. The trial is x_k - h_k (1 + mu) * G / ||G||, with componentwise multiplication denoted by *. The output is [+1, -1, trial]. A tangential gradient norm ||G|| <= 1e-10 is a ValueError.

Unchanged outcome otherwise, including S_k equal to either threshold: return [0, -1, x_k].

```python
def event_proposal(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', normals: 'np.ndarray | list | tuple', supports: 'np.ndarray | list | tuple', target: int, lower: float, upper: float, mu: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Return a length-5 float array [event, neighbor_id, trial_x, trial_y, trial_z].

    event is +1 when the outcome would add a particle, -1 when it would merge
    the inspected particle with a partner, and 0 when the particle set is
    unchanged. neighbor_id is the partner's existing identifier for the
    merging outcome and -1 otherwise. For the unchanged outcome the three
    trial coordinates equal the inspected particle's own position.

    points and normals have shape (n,3) with n>=2; ids has shape (n,) and
    holds unique nonnegative integers, and target is one of them; lengths has
    shape (n,) and is strictly positive; supports has shape (n,) and is
    nonnegative; mu has shape (3,) with every component in [-0.5, 0.5]. Every
    normal has unit norm to absolute tolerance 1e-8. All data and thresholds
    are finite with 0 <= lower < upper. Support exactly equal to either
    threshold selects the unchanged outcome. Ties in any distance-based
    selection are resolved in favour of the smaller existing identifier.
    Any finite-range sum evaluated for the inspected particle takes its range
    from that particle's own characteristic spacing, never from another
    sample's spacing or from any combination of two samples' spacings.

    Raises
    ------
    ValueError
        If any shape, identifier, finiteness, positivity, unit-norm, mu-range
        or threshold condition above fails, if two distinct samples are
        separated by less than 1e-12, or if a proposed trial direction has
        Euclidean norm 1e-10 or less.
    """
    return
```

### Step 5

05_surface_return

Goal
----
Euclidean closest-point return to a positive-axis ellipsoid.

On sum_d (x_d/axes_d)^2 = 1, return the global minimizer of Euclidean distance to trial. The trial may be outside, on, or inside the ellipsoid, including its center. The semiaxes may be unordered or repeated. When multiple points attain the minimum, return the lexicographically greatest point in Cartesian coordinate order. This deterministic convention also applies to symmetric fusion trials.

trial and axes are finite length-three array-like inputs. Every semiaxis and the finite scalar tolerance are strictly positive. tolerance requests absolute positional accuracy in the units of trial. Numerical comparisons use absolute component error 1e-9. Return a finite float array of shape (3,) without changing either input.

Raises ------ ValueError For incorrect input shapes, nonfinite entries, nonpositive axes, or nonpositive/nonfinite tolerance.

```python
def surface_return(trial: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', tolerance: float = 1e-13) -> 'np.ndarray':
    """Return the global closest point under the module's tie convention.

    Interior trials, repeated axes and unordered axes are valid. ValueError
    is required for invalid shapes, nonfinite entries, nonpositive axes,
    or nonpositive/nonfinite tolerance. Both inputs remain unchanged.
    """
    return
```

### Step 6

06_inspected_state

Goal
----
One externally scheduled inspection with current geometric and support fields.

The inspected identifier is an input, not an inferred event scheduler. Apply
this task's support test to fields recomputed from the data supplied in this
call, return any newly created sample to the ellipsoid, and preserve the
ordering of the surviving rows. A later visit must recompute its own fields
from the state this call returns.

```python
def inspected_state(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', h0: float, tau: float, radius: float, lower: float, upper: float, mu: 'np.ndarray | list | tuple', target: int, new_id: int, kref: float=0.0, packing: float=0.933) -> 'np.ndarray':
    """Return an (m,4) numeric array, each row [id, x, y, z], after one inspection.

    Surviving rows keep their incoming order. points has shape (n,3) with n>=2
    and every sample on the ellipsoid; ids holds unique nonnegative integers.
    axes, h0, tau, radius, lower, upper, mu, kref and packing carry the
    meanings and admissible ranges of the earlier primitives. target must be
    present in ids and new_id must be a nonnegative integer not already in
    use. An additive event appends one row carrying new_id; a merging event
    removes the inspected particle and its selected partner and then appends
    one row carrying new_id; an unchanged outcome preserves every row and does
    not consume new_id. Recompute the current fields through the earlier
    subproblem functions rather than reimplementing them.

    Raises
    ------
    ValueError
        If new_id is not an unused nonnegative integer, if any condition of
        the earlier primitives fails on the supplied data, or if the proposal
        reaches a branch those primitives exclude.
    """
    return
```

### Step 7

07_adapted_moment

Goal
----
Final orchestrator: finite local inspection response on a curved cloud.



Generate the complete ellipsoid instance, append the tangent-ring input, run

the three supplied inspections, and obtain the occupied-area-weighted second

moment. The final orchestrator must use the earlier subproblem functions.

```python
def adapted_moment(tau: float=0.5, radius: float=0.55, lower: float=0.7, upper: float=1.25, ring_radius: float=0.12) -> float:
    """Return the final finite scalar sum_i A_i*x_i^2/sum_i A_i.

    Fixed axes are (1,.8,.6), h0=.42, kref=0, packing=.933 and mu=(.13,-.21,.07).
    Generate 48 points: t=1-2(i+.5)/48, phi=i*pi*(3-sqrt(5)),
    x=(sqrt(1-t*t)*cos(phi),.8*sqrt(1-t*t)*sin(phi),.6*t); delete x>.6,z>.1.
    At surviving original ID34, let n be its outward normal,
    e=normalize(n cross (0,0,1)), f=n cross e. Append closest points to
    x34+ring_radius*(cos(2*pi*k/5)*e+sin(2*pi*k/5)*f), k=0,...,4, IDs100+k.
    Inspect IDs [26,34,26], assigning new IDs [200,201,202] by visit if active.
    Re-estimate fields between visits and for the final areas. tau>=0,
    radius>0, ring_radius>0, 0<=lower<upper; all parameters finite. The final
    particle count is not fixed.

    Raises
    ------
    ValueError
        If any parameter is not finite or violates the ranges above, if any
        branch excluded by an earlier primitive is reached, or if an
        identifier a later inspection needs has already been removed.
    """
    return
```
