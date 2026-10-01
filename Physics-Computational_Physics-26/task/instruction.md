# Physics-Computational_Physics-26

## Background

An embedded-boundary or immersed-boundary method keeps a simple Cartesian mesh and
represents a curved solid by cutting through it, rather than fitting a body-conforming
grid. The difficulty moves to the cells the boundary passes through: the discretisation
there needs values at points that lie inside the solid, or close enough to the boundary
that the usual stencil is not available. Ghost cells carry those values, reconstructed
so that the boundary condition is met.

Four ingredients matter here. In three dimensions a trilinear interpolant through seven
Cartesian neighbours together with one boundary point determines a ghost value, and for
the regular geometry of a Cartesian stencil this determination is available in closed
form rather than by assembling and inverting a small local system. What is prescribed at
that boundary point may be the field itself or its derivative along the normal, and the
two cases are not the same closed form. The value-prescribed expression is built from
three coordinate ratios used singly, pairwise and as a triple product; the
derivative-prescribed expression carries seven independent neighbour weights over a
common denominator and cannot be obtained from the first by changing signs, even though
the stencil geometry is identical.

Separately, the stencil of one ghost cell can contain other ghost cells whose values are
themselves unknown, which is conventionally handled by sweeping repeatedly until the
ghost values stop changing. This is where the structure of the problem helps: if the
dependencies among ghost cells contain no cycle, the ghost cells can be arranged so that
each one reads only cells already resolved, and a fixed number of passes replaces the
iteration.

On an anisotropic mesh two further subtleties appear. Any proximity test that decides
which cells become ghost cells has to be measured against a length scale that depends on
the direction of the boundary normal, because the distance a cell spans along that normal
is not the same as any one of the three mesh spacings. And where a configuration keeps
more than one layer of ghost cells, the deeper ones sit too far from their boundary point
for the nearest-neighbour stencil to remain well conditioned, so the stencil is stretched
by a whole number of cells independently in each coordinate direction. The extra layer
and the stretched stencil are a pair: neither is useful without the other.

## Problem

Embedded-boundary methods place a curved solid inside a Cartesian mesh and recover the
solution in the cut cells by reconstructing values at ghost cells. A recent line of work
gives a direct, non-iterative construction for those ghost values in three dimensions,
together with the rule that decides which cells become ghost cells and, where a cell sits
too deep for the nearest-neighbour stencil, the rule that widens that stencil. Your task is
to run that construction on one prescribed configuration and report how accurately it
reconstructs a known field.

Use the following configuration.

- Mesh: 31 cell-centre coordinates in each direction, endpoints included, so that
  x_i = -1.5 + 0.100 i, y_j = -1.2 + 0.080 j and z_k = -0.9 + 0.060 k for
  i, j, k = 0, 1, ..., 30. The three spacings differ.
- Embedded boundary: a sphere of radius 0.61 centred at (0.07, -0.11, 0.05). The ball
  interior is solid and everything outside it is fluid.
- Known field: phi(x, y, z) = sin(1.3 x) cos(0.9 y) sin(0.7 z) + 0.45 x y z, supplied as the
  surrounding solution at fluid cells. Ghost values are set to zero before reconstruction.
- Boundary data is mixed. Where the outward unit normal at the boundary point has a
  non-negative x component, the normal derivative grad(phi) . n is prescribed there, n
  being that same outward unit normal. On the remainder of the sphere phi itself is
  prescribed there instead.
- The boundary point for a cell is where the boundary normal through that cell centre
  meets the sphere, with the normal oriented into the fluid.
- Ghost cells: the configuration keeps a second layer of them on the solid side, using the
  deeper of the source's two proximity thresholds there and the shallower one on the fluid
  side, and it widens the reconstruction stencil wherever the source's extended-stencil
  rule requires it. That second layer is kept in full, so the source's excess-candidate
  reclassification applies to the fluid side only, with a Cartesian neighbour counted as
  solid whenever its centre lies inside the ball, including when it is retained as a
  solid-side ghost.

Reconstruct every ghost value and form the absolute difference between the reconstructed
ghost values and the known field at those same cells. In your reasoning, report the mean
of those absolute differences, together with four integers that characterise the
configuration: how many cells are tagged as ghost cells, how many of those ghost centres
lie in the fluid region, how many of them have a widened stencil in at least one
direction, and how many ordering passes the reconstruction needs. State as well, in one
line each, the algebraic form of the interpolant both reconstructions are derived from,
how the derivative-prescribed reconstruction differs in structure from the
value-prescribed one, and the rule that fixes how far the stencil is widened. Your final
answer must be a single number: the largest of those absolute differences.

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

signed_normal_distance

Goal
----
Return the signed distance from each cell centre to the spherical embedded boundary, measured along the boundary normal through that centre. The sign convention is positive on the solid side and negative on the fluid side. The solid is the ball interior and the fluid is everything outside it.

```python
def signed_normal_distance(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    """Return the signed distance from each cell centre to the spherical embedded boundary, measured along the boundary normal through that centre. The sign convention is positive on the solid side and negative on the fluid side. The solid is the ball interior and the fluid is everything outside it.

    Returns
    -------
    ndarray with the same shape as xc, float64: the signed normal distance.
    """
    return result
```

### Step 2

boundary_normal

Goal
----
Return the unit normal to the spherical embedded boundary at each cell centre, oriented to point out of the solid and into the fluid region.

```python
def boundary_normal(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float) -> "np.ndarray":
    """Return the unit normal to the spherical embedded boundary at each cell centre, oriented to point out of the solid and into the fluid region.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the x, y and z components of the unit normal.
    """
    return result
```

### Step 3

normal_cell_thickness

Goal
----
Return the characteristic cell thickness measured in the boundary-normal direction. Divide each component of the unit normal by the mesh spacing in that same coordinate direction, take the Euclidean norm of the resulting triple, and return its reciprocal. It equals the spacing exactly when the normal is aligned with that axis and lies between the smallest and largest spacing otherwise, so it is not simply the smallest of the three.

```python
def normal_cell_thickness(normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the characteristic cell thickness measured in the boundary-normal direction. Divide each component of the unit normal by the mesh spacing in that same coordinate direction, take the Euclidean norm of the resulting triple, and return its reciprocal. It equals the spacing exactly when the normal is aligned with that axis and lies between the smallest and largest spacing otherwise, so it is not simply the smallest of the three.

    Returns
    -------
    ndarray with the shape of one normal component, float64: the thickness.
    """
    return result
```

### Step 4

hybrid_ghost_tags

Goal
----
Tag every cell as fluid, solid, or a hybrid ghost cell. A cell whose signed distance is positive is a ghost candidate when that distance is at most one and a half times its thickness, which keeps a second layer of them on the solid side; a cell whose signed distance is negative is a ghost candidate when the distance measured into the fluid is strictly less than half its thickness. Then make one follow-up pass over the fluid-side candidates only: a fluid-side candidate with no solid cell among its six Cartesian neighbours becomes plain fluid. Solid-side candidates are all retained, because the deeper threshold admitted them deliberately. Both the candidacy tests and this follow-up pass classify neighbours by the sign of the signed distance, never by a tag this same call has already written. Cells at an array face have their off-grid neighbours replaced by the face cell itself.

```python
def hybrid_ghost_tags(psi: "np.ndarray", thickness: "np.ndarray") -> "np.ndarray":
    """Tag every cell as fluid, solid, or a hybrid ghost cell. A cell whose signed distance is positive is a ghost candidate when that distance is at most one and a half times its thickness, which keeps a second layer of them on the solid side; a cell whose signed distance is negative is a ghost candidate when the distance measured into the fluid is strictly less than half its thickness. Then make one follow-up pass over the fluid-side candidates only: a fluid-side candidate with no solid cell among its six Cartesian neighbours becomes plain fluid. Solid-side candidates are all retained, because the deeper threshold admitted them deliberately. Both the candidacy tests and this follow-up pass classify neighbours by the sign of the signed distance, never by a tag this same call has already written. Cells at an array face have their off-grid neighbours replaced by the face cell itself.

    Returns
    -------
    ndarray with the shape of psi, float64: 0.0 fluid, 1.0 solid, 2.0 a ghost whose centre lies in the solid, -2.0 a ghost whose centre lies in the fluid.
    """
    return result
```

### Step 5

boundary_point

Goal
----
Return the point where the boundary normal through each cell centre meets the spherical embedded boundary. This is the location at which the boundary condition is imposed.

```python
def boundary_point(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", centre_x: float, centre_y: float, centre_z: float, radius: float) -> "np.ndarray":
    """Return the point where the boundary normal through each cell centre meets the spherical embedded boundary. This is the location at which the boundary condition is imposed.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 hold the x, y and z coordinates of the intersection.
    """
    return result
```

### Step 6

stencil_multipliers

Goal
----
Return the per-direction integer by which the reconstruction stencil is widened at each cell. For each coordinate direction take twice the absolute value of the product of that direction's normal component with the signed distance, divide by that direction's mesh spacing, round up to the next integer, and take the larger of that and one. The multiplier is therefore one wherever the normal component vanishes or the cell sits close to the boundary.

```python
def stencil_multipliers(psi: "np.ndarray", normals: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the per-direction integer by which the reconstruction stencil is widened at each cell. For each coordinate direction take twice the absolute value of the product of that direction's normal component with the signed distance, divide by that direction's mesh spacing, round up to the next integer, and take the larger of that and one. The multiplier is therefore one wherever the normal component vanishes or the cell sits close to the boundary.

    Returns
    -------
    ndarray of shape (3,) + psi.shape, float64: entries 0, 1 and 2 are the multipliers in the x, y and z directions.
    """
    return result
```

### Step 7

trilinear_value_weights

Goal
----
Return the three interpolation weights of the closed-form ghost-cell reconstruction that applies where the field value itself is prescribed at the boundary. The diagonally opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side. Each weight is then the ghost-cell coordinate minus the boundary-point coordinate, divided by the opposite stencil point's coordinate minus the boundary-point coordinate, taken in x for the first weight, in y for the second and in z for the third.

```python
def trilinear_value_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the three interpolation weights of the closed-form ghost-cell reconstruction that applies where the field value itself is prescribed at the boundary. The diagonally opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side. Each weight is then the ghost-cell coordinate minus the boundary-point coordinate, divided by the opposite stencil point's coordinate minus the boundary-point coordinate, taken in x for the first weight, in y for the second and in z for the third.

    Returns
    -------
    ndarray of shape (3,) + xc.shape, float64: entries 0, 1 and 2 are the weights built from the x, y and z coordinates.
    """
    return result
```

### Step 8

trilinear_derivative_weights

Goal
----
Return the eight weights of the closed-form ghost-cell reconstruction that applies where the normal derivative is prescribed at the boundary: seven on the Cartesian neighbours, in the P1 to P7 order fixed below, and one on that prescribed derivative. The weights are defined by the interpolation problem itself. Over the stencil formed by the ghost cell and its seven Cartesian neighbours there is a unique function of the form C0 + C1 x + C2 y + C3 z + C4 xy + C5 yz + C6 xz + C7 xyz whose values at the seven neighbours are the seven neighbour values and whose gradient dotted with the outward unit normal equals the prescribed derivative at the boundary point. That function's value at the ghost cell depends linearly on those eight data; return the eight coefficients of that dependence, the seven neighbour coefficients first and the coefficient of the prescribed derivative last. The opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side, which is what keeps the problem non-degenerate. Writing i2, j2 and k2 for that point's indices in x, y and z, the seven neighbours in order are the points differing from the ghost cell in x alone, in y alone, in z alone, in x and y, in y and z, in x and z, and in all three.

```python
def trilinear_derivative_weights(xc: "np.ndarray", yc: "np.ndarray", zc: "np.ndarray", boundary: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray", dx: float, dy: float, dz: float) -> "np.ndarray":
    """Return the eight weights of the closed-form ghost-cell reconstruction that applies where the normal derivative is prescribed at the boundary: seven on the Cartesian neighbours, in the P1 to P7 order fixed below, and one on that prescribed derivative. The weights are defined by the interpolation problem itself. Over the stencil formed by the ghost cell and its seven Cartesian neighbours there is a unique function of the form C0 + C1 x + C2 y + C3 z + C4 xy + C5 yz + C6 xz + C7 xyz whose values at the seven neighbours are the seven neighbour values and whose gradient dotted with the outward unit normal equals the prescribed derivative at the boundary point. That function's value at the ghost cell depends linearly on those eight data; return the eight coefficients of that dependence, the seven neighbour coefficients first and the coefficient of the prescribed derivative last. The opposite stencil point sits one multiplier's worth of cells away in each direction, on the side the matching normal component points to, counting a vanishing component as the positive side, which is what keeps the problem non-degenerate. Writing i2, j2 and k2 for that point's indices in x, y and z, the seven neighbours in order are the points differing from the ghost cell in x alone, in y alone, in z alone, in x and y, in y and z, in x and z, and in all three.

    Returns
    -------
    ndarray of shape (8,) + xc.shape, float64: entries 0 to 6 are the seven neighbour weights in the order the description lists them, matching stencil points P1 to P7, and entry 7 is the weight on the prescribed normal derivative.
    """
    return result
```

### Step 9

dependency_levels

Goal
----
Assign every ghost cell to a dependency level, so that the reconstruction of a ghost cell in one level needs only cells that are already resolved. A ghost cell is one tagged 2.0 or -2.0; every other cell carries -1.0. A ghost cell whose active stencil neighbours contain no other ghost cell is at level 0; otherwise its level is one more than the largest level among the ghost cells in its active stencil. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. Write i2, j2 and k2 for the index one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds. The seven stencil neighbours are then (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2) and (k2,j2,i2). Omit a neighbour whenever reaching it changes a coordinate whose normal component is zero: its reconstruction weight vanishes, so it creates no dependency. Memoise, because the same cell is reached from many others. Raise ValueError if the dependencies contain a cycle, which an active stencil dependency that clamps back onto its own cell produces.

```python
def dependency_levels(tags: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    """Assign every ghost cell to a dependency level, so that the reconstruction of a ghost cell in one level needs only cells that are already resolved. A ghost cell is one tagged 2.0 or -2.0; every other cell carries -1.0. A ghost cell whose active stencil neighbours contain no other ghost cell is at level 0; otherwise its level is one more than the largest level among the ghost cells in its active stencil. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. Write i2, j2 and k2 for the index one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds. The seven stencil neighbours are then (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2) and (k2,j2,i2). Omit a neighbour whenever reaching it changes a coordinate whose normal component is zero: its reconstruction weight vanishes, so it creates no dependency. Memoise, because the same cell is reached from many others. Raise ValueError if the dependencies contain a cycle, which an active stencil dependency that clamps back onto its own cell produces.

    Returns
    -------
    ndarray with the shape of tags, float64: the level index of each ghost cell as a float, counting from 0.0, and -1.0 at every cell that is not a ghost.
    """
    return result
```

### Step 10

sweep_reconstruct

Goal
----
Reconstruct every ghost value by visiting the dependency levels in increasing order, writing the values of one level before moving to the next. Within a level the ghost cells are independent of one another. Cells that are not ghosts are left untouched. A ghost cell is one tagged 2.0 or -2.0. The stencil is the general eight-term one: the ghost value is the sum over the seven stencil neighbours of that neighbour's weight times its current value, plus the eighth weight, so the caller supplies whichever closed form applies at each cell rather than the routine assuming one. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. With i2, j2 and k2 the indices one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds, the seven neighbours are taken in the order (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2), (k2,j2,i2).

```python
def sweep_reconstruct(field: "np.ndarray", tags: "np.ndarray", levels: "np.ndarray", weights: "np.ndarray", normals: "np.ndarray", mult: "np.ndarray") -> "np.ndarray":
    """Reconstruct every ghost value by visiting the dependency levels in increasing order, writing the values of one level before moving to the next. Within a level the ghost cells are independent of one another. Cells that are not ghosts are left untouched. A ghost cell is one tagged 2.0 or -2.0. The stencil is the general eight-term one: the ghost value is the sum over the seven stencil neighbours of that neighbour's weight times its current value, plus the eighth weight, so the caller supplies whichever closed form applies at each cell rather than the routine assuming one. Index the arrays [k, j, i] with k along z, j along y and i along x, so entry 0 of normals and mult belongs to the LAST index and entry 2 to the FIRST. With i2, j2 and k2 the indices one multiplier's worth of cells away in x, y and z, on the side the matching normal component points to, counting a vanishing component as the positive side, and clamped to the array bounds, the seven neighbours are taken in the order (k,j,i2), (k,j2,i), (k2,j,i), (k,j2,i2), (k2,j2,i), (k2,j,i2), (k2,j2,i2).

    Returns
    -------
    ndarray with the shape of field, float64: a copy of field with every ghost cell replaced by its reconstructed value.
    """
    return result
```

### Step 11

embedded_boundary_audit

Goal
----
Run the whole audit on the prescribed instance at the given refinement by chaining the earlier steps, and return the eight diagnostics that characterise it. The instance is the one the problem statement fixes; refine = 1 is that instance and larger integers subdivide each direction by that factor while keeping the same extents and the same sphere.

```python
def embedded_boundary_audit(refine: int) -> "np.ndarray":
    """Run the whole audit on the prescribed instance at the given refinement by chaining the earlier steps, and return the eight diagnostics that characterise it. The instance is the one the problem statement fixes; refine = 1 is that instance and larger integers subdivide each direction by that factor while keeping the same extents and the same sphere.

    Returns
    -------
    ndarray of shape (8,), float64: the largest absolute ghost-cell reconstruction error, the mean absolute error, the number of ghost cells, the number of ordering passes, the number of derivative-type ghost cells, the smallest normal-direction thickness over the ghost cells, the number of ghost cells with a widened stencil in at least one direction, and the number of ghost centres lying in the fluid.
    """
    return result
```
