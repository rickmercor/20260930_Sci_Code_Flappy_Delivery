# Mathematics-Computational_Mechanics-4

## Background

Peridynamics is a nonlocal reformulation of continuum solid mechanics in which the spatial derivatives of classical elasticity are replaced by integral operators over a finite neighbourhood. Because the governing equations never differentiate the displacement field, discontinuities such as cracks are admissible solutions rather than singularities requiring special treatment, and crack initiation, propagation, branching, and arrest emerge from the constitutive model itself without an external crack-growth law. This makes the framework attractive for dynamic brittle fracture, where classical methods need enrichment functions, remeshing, or a separate phase-field variable to represent evolving discontinuities.

In the bond-based formulation, each material point $\boldsymbol{X}_i$ interacts with every point inside a spherical neighbourhood of radius $\delta$, the horizon, and the interaction along each such bond depends only on that bond's own deformation. The bond stretch $s_{ij} = (\lVert\boldsymbol{x}_{ij}\rVert - \lVert\boldsymbol{X}_{ij}\rVert)/\lVert\boldsymbol{X}_{ij}\rVert$ measures its strain, and a micro-potential quadratic in $s_{ij}$ recovers linear elastic behaviour in the bulk. The constant multiplying that micro-potential is not a free parameter: it is fixed by requiring that the nonlocal strain energy density reproduce the classical value under a reference homogeneous deformation, which ties the model to measurable elastic constants and makes it horizon-dependent. Bond-based models pay for their simplicity with a constrained Poisson's ratio, but remain widely used for fracture because of their conceptual clarity and low cost per bond.

Practical fracture simulations rarely warrant uniform resolution. High resolution is needed only near crack tips and along expected crack paths, while the remainder of the domain can be discretised far more coarsely. Since the horizon is conventionally set proportional to the local point spacing, local mesh refinement makes the horizon a spatially varying field. This introduces a difficulty absent from the uniform case: neighbourhood membership stops being symmetric, so a point may lie inside its neighbour's horizon without that neighbour lying inside its own. Naive pairwise summation over each point's own neighbourhood then produces unbalanced interactions, which manifest as ghost forces and as spurious reflection of stress waves at resolution interfaces even though no material discontinuity is present there. Reported consequences include artificially deflected crack paths, as though the refined region were bounded by an impenetrable barrier.

A parallel difficulty arises in time. Explicit integration of peridynamic dynamics is conditionally stable, with a critical step size that scales with the local horizon and discretisation, so a globally uniform step must be chosen to satisfy the most restrictive region in the domain. Where refinement is localised, this forces the entire mesh to advance at the pace of its smallest cells, and most of the resulting internal-force evaluations do no useful work. Multirate and asynchronous schemes address this by giving each interaction potential its own update grid and merging all local update times into a single ordered global sequence, so that coarse regions are updated less often than fine ones while the coupling between them remains consistent.

Both difficulties are addressed more cleanly by deriving the discretisation from a variational principle than by patching the equations after the fact. Starting from the Lagrange–d'Alembert principle and discretising the action rather than the resulting differential equations yields integrators that are symplectic, satisfy a discrete Noether theorem, and therefore conserve linear and angular momentum exactly while exhibiting no secular energy drift over long integration times. Different quadrature rules applied to the same discrete action generate a hierarchy of schemes of differing accuracy, and the same variational route accommodates asynchronous update grids directly. This structure-preserving character is not decorative for fracture problems: momentum and energy bookkeeping is what distinguishes genuine crack dynamics from artefacts of the discretisation, particularly at interfaces between regions of differing resolution.

Fracture itself enters through irreversible bond failure. Each bond carries a history-dependent Boolean state; once the bond reaches the threshold set by the chosen failure criterion it is removed permanently and transmits no further force, and the accumulated fraction of broken bonds at a point defines a local damage measure whose contours trace the crack. The threshold is calibrated by equating the total work required to break every bond crossing a unit fracture surface to the material's fracture energy $G_c$, so that the nonlocal model dissipates the correct energy per unit of created crack area. How that calibration is carried out, and how it interacts with a horizon that varies in space, is central to whether a peridynamic fracture simulation is quantitatively meaningful.

## Problem

Bond-based peridynamics with locally refined discretisations gives each material point a horizon proportional to its local grid spacing, so the horizon becomes a spatially varying field. Neighbourhood membership is then no longer reciprocal, and naive pairwise force summation violates the balance laws, producing ghost forces and spurious wave reflections at refinement interfaces. Deriving the equations of motion variationally from the Lagrange–d'Alembert principle restores exact momentum balance and further yields structure-preserving multirate integrators that admit different time step sizes in different regions of the domain.

Your task is to solve one concrete deterministic example of this pipeline. A pre-cracked plane-strain brittle plate is discretised into two regions of different resolution, and internal forces follow the variational, momentum-conserving formulation for spatially varying horizons. Each bond carries the micro-potential $\pi^{\mathrm{int}}_{ij} = \tfrac{1}{2}\,c\,\omega(\lVert \boldsymbol{X}_{ij}\rVert)\,s_{ij}^{2}\,\lVert \boldsymbol{X}_{ij}\rVert$, and $c$ is fixed by requiring that the total internal energy of the discrete body under homogeneous isotropic extension, summed over every point and over every member of that point's family, reproduce the classical plane-strain strain energy density. A bond fails irreversibly once the energy density stored in it reaches the critical value obtained by equating the total work needed to break all bonds crossing a unit fracture area to the fracture energy $G_c$. Time integration uses the multirate variational scheme obtained from trapezoidal quadrature of the discrete action, with the two regions advanced on aligned local time grids. Use the following configuration (SI units, all quantities per unit out-of-plane thickness):

- Material: $E = 72$ GPa, $\varrho = 2440$ kg/m$^3$, $G_c = 135$ J/m$^2$, plane strain
- Influence function: $\omega(r) = r^{-2}$
- Domain $[0, 0.04] \times [0, 0.02]$ m; region I ($x < 0.02$): cell-centred $10 \times 10$ lattice with spacing $\Delta_1 = 0.002$; region II ($x > 0.02$): cell-centred $20 \times 20$ lattice with spacing $\Delta_2 = 0.001$
- Horizons $\delta = 3.015\,\Delta$ within each region; the family of a point contains every other point within that point's own horizon; voxel volumes are $\Delta^2$, with no partial-volume or surface corrections
- The horizon entering any per-bond quantity is the one that defines that bond's micro-potential
- Pre-crack: every bond whose reference segment crosses the line $y = 0.0101$ m, $x \le 0.0151$ m is removed from the families
- Loading: constant body-force density $b_y = \pm\,\sigma/\Delta_{\mathrm{local}}$ with $\sigma = 14$ MPa on the single outermost row of points along the top ($+$) and bottom ($-$) edges, active from $t = 0$; initial displacements and velocities are zero
- No-fail layers: bonds with either endpoint at $y < 0.0021$ m or $y > 0.0179$ m never fail
- Bond states are updated at the owning potential's local update events, from the current deformed configuration, immediately before that event's force contribution is computed
- Local time steps $h_1 = 4 \times 10^{-8}$ s (region I) and $h_2 = 2 \times 10^{-8}$ s (region II), both grids starting at $t = 0$; final time $T = 2 \times 10^{-5}$ s
- The explicit stability bound of a region is evaluated from the total stiffness accumulated at each of its points, counting every bond that touches the point

Your final answer must be a single number: the vertical displacement $u_y$, in micrometres, at time $T$, of the material point initially at $(0.0255, 0.0155)$ m.

Report evidence that the formulation conserves linear momentum under the given discretisation, and establish whether the failure criterion specified above is equivalent to one calibrated as a fixed threshold on bond stretch.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> under 1500 words. Show the intermediate quantities that justify the final number, covering the calibration, the discretisation, the failure threshold, and the time integration.

Do not paste input coordinate lists, per-event logs, per-bond tables, or full trajectories.

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

01_build_point_lattice

Goal
----
Construct the cell-centred material-point lattice for a horizontally partitioned rectangular plate whose regions may use different uniform grid spacings.

The domain is the rectangle [x_bounds[0,0], x_bounds[-1,1]] x [0, height]. It is divided into R contiguous vertical regions; region r spans x in [x_bounds[r,0], x_bounds[r,1]] and is discretised with the uniform spacing spacings[r] in both coordinate directions. Within a region, points sit at the centres of square cells of side spacings[r], so the point at cell index (ix, iy) has coordinates (x_lo + (ix + 0.5) * d, (iy + 0.5) * d) with d = spacings[r]. Each point carries its local spacing d, its horizon horizon_ratio * d, and its voxel volume d ** 2 (per unit out-of-plane thickness).

Points are emitted in a fixed order: regions in the order given by x_bounds, then within each region ix ascending as the outer loop and iy ascending as the inner loop. Every downstream step indexes points by their position in this ordering, so the ordering is part of the contract.

The function returns one float array with five columns [x, y, spacing, horizon, volume], one row per point.

Raises ValueError if: x_bounds does not have shape (R, 2) with R >= 1; spacings does not have shape (R,) matching x_bounds; x_bounds or spacings contain non-finite values; any region has x_hi <= x_lo; consecutive regions are not contiguous in x within an absolute tolerance of 1e-12; any spacing is not strictly positive; height is not finite and strictly positive; horizon_ratio is not finite and strictly positive; any region width is not a positive integer multiple of that region's spacing within a relative tolerance of 1e-12; height is not a positive integer multiple of every region's spacing within a relative tolerance of 1e-12.

```python
import numpy as np


def build_point_lattice(x_bounds: np.ndarray, spacings: np.ndarray,
                        height: float, horizon_ratio: float) -> np.ndarray:
    '''Build the cell-centred material-point lattice of a multi-region plate.

    Parameters
    ----------
    x_bounds : np.ndarray
        (R, 2) float array; row r gives the x-interval [x_lo, x_hi] of region r.
        Regions must be contiguous and ordered by increasing x.
    spacings : np.ndarray
        (R,) float array of uniform grid spacings, one per region.
    height : float
        Extent of the plate in the y-direction, spanning [0, height].
    horizon_ratio : float
        Ratio m in delta = m * spacing, applied within each region.

    Returns
    -------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume],
        ordered by region, then by ix ascending, then by iy ascending.
    '''
    lattice = np.zeros((0, 5), dtype=float)
    return lattice  # placeholder
```

### Step 2

02_build_directed_bonds

Goal
----
Enumerate the directed bonds of the peridynamic discretisation and remove those severed by the pre-crack.

Given the lattice produced by the previous step, point i owns a bond to point j whenever j is distinct from i and lies inside i's own horizon, that is ||X_j - X_i|| <= lattice[i, 3] within a relative tolerance of 1e-12. Because horizons vary in space this relation is not symmetric: a bond (i, j) may exist while (j, i) does not.

A bond is severed by the pre-crack when its reference segment crosses the horizontal line y = crack_y at an abscissa not exceeding crack_x_max. The segment crosses when (y_i - crack_y) * (y_j - crack_y) < 0, so a bond with an endpoint lying exactly on the crack line does not cross it; the crossing abscissa is then obtained by linear interpolation along the segment. Severed bonds are omitted from the output entirely, in both directions in which they appear.

Bonds are emitted in a fixed order: owner index ascending, and for a given owner, neighbour index ascending. Every downstream step indexes bonds by their position in this ordering, so the ordering is part of the contract.

The function returns one integer array with two columns [owner, neighbour], one row per surviving directed bond.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; any horizon in column 3 is not strictly positive; any volume in column 4 is not strictly positive; crack_y or crack_x_max is not finite; the lattice contains two coincident points.

```python
import numpy as np


def build_directed_bonds(lattice: np.ndarray, crack_y: float,
                         crack_x_max: float) -> np.ndarray:
    '''Enumerate directed peridynamic bonds, omitting those severed by the pre-crack.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    crack_y : float
        Ordinate of the horizontal pre-crack line.
    crack_x_max : float
        Largest abscissa at which the pre-crack severs a bond.

    Returns
    -------
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour], ordered by owner
        ascending and then by neighbour ascending.
    '''
    bonds = np.zeros((0, 2), dtype=np.int64)
    return bonds  # placeholder
```

### Step 3

03_calibrate_micromodulus

Goal
----
Calibrate the bond micro-modulus constant against the classical plane-strain strain energy density.

The bond micro-potential is quadratic in the bond stretch, pi_ij = 0.5 * c * omega(||X_ij||) * s_ij ** 2 * ||X_ij||, where omega is the influence function and c is a constant that depends on the horizon. The constant is fixed by requiring that, for a uniform horizon and in the bulk of the body, the total internal energy under a homogeneous isotropic extension equal the classical plane-strain value for a material whose Poisson's ratio takes the value imposed by the bond-based formulation. That total is the double sum over every point and every member of that point's family, in which each bond therefore appears twice, and the calibration accounts for both appearances.

The influence function is the power law omega(r) = r ** (-influence_exponent), so the calibration reduces to a single weighted moment of the influence function over the neighbourhood, evaluated in polar coordinates. The moment converges only for influence_exponent < 3.

The function is evaluated for an array of horizons and returns an array of the same shape, giving the constant for each. The returned value is the constant appearing in the micro-potential itself and is used unchanged wherever that micro-potential or its derivative is required.

Raises ValueError if: horizons is empty; horizons contains non-finite values; any horizon is not strictly positive; youngs_modulus is not finite and strictly positive; influence_exponent is not finite; influence_exponent is greater than or equal to 3.

```python
import numpy as np


def calibrate_micromodulus(youngs_modulus: float, horizons: np.ndarray,
                           influence_exponent: float) -> np.ndarray:
    '''Calibrate the plane-strain bond micro-modulus for each given horizon.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus E of the material.
    horizons : np.ndarray
        Array of horizon values, of any shape.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    micromodulus : np.ndarray
        Array of the same shape as horizons, giving the constant appearing in
        the micro-potential for each horizon.
    '''
    micromodulus = np.zeros_like(np.asarray(horizons, dtype=float))
    return micromodulus  # placeholder
```

### Step 4

04_bond_critical_stretch_squared

Goal
----
Compute the squared critical stretch of every directed bond from the critical energy density.

A bond fails once the energy density stored in it reaches a critical value w_c. That critical value is fixed by equating the total work required to break every bond crossing a unit fracture area to the fracture energy G_c, an integration that in two dimensions depends only on the horizon and yields a w_c independent of the influence function.

Because the stored energy density is quadratic in the bond stretch, the criterion is equivalent to a threshold on the squared stretch, and that threshold depends on the individual bond through both its reference length and the influence function evaluated at that length. The function returns this threshold, not the energy density itself, so that the failure test downstream is a direct comparison against the squared stretch.

The energy a bond stores is a physical quantity and does not depend on how the internal energy of the body is written as a sum. The micromodulus supplied to this function is the constant appearing in the micro-potential of the double-sum form, in which each bond appears twice, so the energy stored in one bond is obtained by accounting for both appearances before the threshold is formed.

The horizon and micromodulus used for a given bond are those of the point that parameterises that bond's micro-potential, consistent with the convention used everywhere else in the pipeline; for a spatially varying horizon these differ from the corresponding quantities of the bond's owner.

The influence function is the power law omega(r) = r ** (-influence_exponent). Bonds are indexed as in the directed bond list, and the returned array has one entry per directed bond, in that same order.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; fracture_energy is not finite and strictly positive; influence_exponent is not finite; influence_exponent is greater than or equal to 3; any bond has non-positive reference length.

```python
import numpy as np


def bond_critical_stretch_squared(lattice: np.ndarray, bonds: np.ndarray,
                                  micromodulus: np.ndarray, fracture_energy: float,
                                  influence_exponent: float) -> np.ndarray:
    '''Squared critical stretch of each directed bond under the critical energy density criterion.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    fracture_energy : float
        Critical energy release rate G_c of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    critical_stretch_squared : np.ndarray
        (B,) float array giving the squared critical stretch of each directed
        bond, in the bond ordering.
    '''
    critical_stretch_squared = np.zeros(len(np.asarray(bonds)), dtype=float)
    return critical_stretch_squared  # placeholder
```

### Step 5

05_regional_critical_time_step

Goal
----
Estimate the critical explicit time step of each region from the discrete peridynamic stiffness.

Von Neumann stability analysis of the explicit update bounds the step size at a material point by the square root of twice the mass density divided by the total stiffness accumulated at that point. The stiffness assembled here is that of the equations of motion the integrator actually advances, so every directed bond contributes to both of the points it joins, using the same coefficient that multiplies the stretch in the pairwise force and the same volume weighting: the contribution deposited on a bond's owner carries the neighbour's volume, and the contribution deposited on its neighbour carries the owner's volume. Under a spatially varying horizon these two contributions differ, so a point's stiffness cannot be obtained from its own family alone.

This convention differs from the form in which the bound is usually tabulated, where a single summation over each point's own family is taken; that form is equivalent only when neighbourhood membership is reciprocal.

The critical step of a region is the smallest point-wise bound among the points assigned to that region.

Points are assigned to regions by region_ids. The returned array has one entry per distinct identifier present, ordered by increasing identifier.

The influence function is the power law omega(r) = r ** (-influence_exponent), and micromodulus is the constant appearing in the micro-potential, as returned by the calibration step and used unchanged.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2) with B >= 1; any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; region_ids does not have shape (N,) matching lattice; region_ids is not an integer array; density is not finite and strictly positive; influence_exponent is not finite or is greater than or equal to 3; any bond has non-positive reference length; any point accumulates no stiffness contribution.

```python
import numpy as np


def regional_critical_time_step(lattice: np.ndarray, bonds: np.ndarray,
                                micromodulus: np.ndarray, region_ids: np.ndarray,
                                density: float, influence_exponent: float) -> np.ndarray:
    '''Critical explicit time step of each region.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    region_ids : np.ndarray
        (N,) integer array assigning each point to a region.
    density : float
        Mass density of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    critical_step : np.ndarray
        Array with one entry per distinct region identifier, ordered by
        increasing identifier.
    '''
    critical_step = np.zeros(len(np.unique(np.asarray(region_ids))), dtype=float)
    return critical_step  # placeholder
```

### Step 6

06_internal_force_density

Goal
----
Accumulate the internal peridynamic force density at every material point for a given displacement field and bond state.

Each directed bond contributes a central pairwise force along its deformed direction, obtained by differentiating that bond's micro-potential with respect to the deformed bond vector, so the coefficient multiplying the stretch is the micro-potential constant supplied to this function, used unchanged. Broken bonds, whose state is zero, contribute nothing.

Because the internal potential is a double sum over points and their families, its variation places two contributions on every directed bond: one on the bond's owner and one on its neighbour, equal in magnitude and opposite in direction as force densities, but weighted by different quadrature volumes because the two points occupy voxels of different size. A point therefore accumulates contributions both from the bonds it owns and from the bonds that reach it as a neighbour, and neither set may be omitted when horizons vary in space.

The returned quantity is a force density, that is, the internal term of mass density times acceleration; it does not include any external body force. It has one row per material point, in the lattice ordering.

The influence function is the power law omega(r) = r ** (-influence_exponent), and micromodulus is the constant appearing in the micro-potential, as returned by the calibration step.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; bond_states does not have shape (B,) matching bonds; bond_states contains a value other than 0 or 1; displacement does not have shape (N, 2) matching lattice; displacement contains non-finite values; influence_exponent is not finite or is greater than or equal to 3; any bond has non-positive reference length; any bond has non-positive deformed length.

```python
import numpy as np


def internal_force_density(lattice: np.ndarray, bonds: np.ndarray,
                           micromodulus: np.ndarray, bond_states: np.ndarray,
                           displacement: np.ndarray,
                           influence_exponent: float) -> np.ndarray:
    '''Accumulate internal force density at every point.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    bond_states : np.ndarray
        (B,) array of bond states, 1 for intact and 0 for broken.
    displacement : np.ndarray
        (N, 2) float array of current displacements.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    force : np.ndarray
        (N, 2) float array of internal force density at each point, in the
        lattice ordering.
    '''
    force = np.zeros((len(np.asarray(lattice)), 2), dtype=float)
    return force  # placeholder
```

### Step 7

07_update_bond_states

Goal
----
Apply the irreversible bond-failure criterion to the current deformed configuration and return the updated bond states.

For every directed bond the current stretch is computed from the deformed positions, and the bond fails when the square of that stretch reaches or exceeds the squared critical stretch supplied for it. Comparing squares rather than signed stretches is what the energy-based criterion requires, since the energy stored in a bond does not depend on the sign of its deformation.

Failure is irreversible: a bond whose incoming state is zero remains zero regardless of its current stretch, and is never restored.

A bond is exempt from failure when either of the points it joins is marked in no_fail_points. Exempt bonds retain their incoming state.

Bond states are returned in the bond ordering, with the value 1 for intact and 0 for broken. The incoming array is not modified.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; bond_states does not have shape (B,) matching bonds; bond_states contains a value other than 0 or 1; critical_stretch_squared does not have shape (B,) matching bonds; critical_stretch_squared contains non-finite or non-positive values; displacement does not have shape (N, 2) matching lattice; displacement contains non-finite values; no_fail_points does not have shape (N,) matching lattice; no_fail_points contains a value other than 0 or 1; any bond has non-positive reference length; any bond has non-positive deformed length.

```python
import numpy as np


def update_bond_states(lattice: np.ndarray, bonds: np.ndarray, bond_states: np.ndarray,
                       critical_stretch_squared: np.ndarray, displacement: np.ndarray,
                       no_fail_points: np.ndarray) -> np.ndarray:
    '''Apply the irreversible failure criterion and return updated bond states.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    bond_states : np.ndarray
        (B,) array of incoming bond states, 1 for intact and 0 for broken.
    critical_stretch_squared : np.ndarray
        (B,) float array of squared critical stretch, one per directed bond.
    displacement : np.ndarray
        (N, 2) float array of current displacements.
    no_fail_points : np.ndarray
        (N,) array marking points exempt from failure, 1 for exempt and 0 otherwise.

    Returns
    -------
    updated_states : np.ndarray
        (B,) float array of updated bond states, in the bond ordering.
    '''
    updated_states = np.asarray(bond_states, dtype=float).copy()
    return updated_states  # placeholder
```

### Step 8

08_integrate_avv

Goal
----
Advance the peridynamic body to the final time with the asynchronous variational velocity-Verlet scheme, updating bond states as the integration proceeds.

Each material point's interaction potential carries its own local update grid, whose spacing is given by local_steps. Local grids all begin at time zero. The global grid is the union of the local grids, and every global node advances positions and velocities for the whole body.

Force contributions are exchanged as impulses rather than applied instantaneously. When a potential reaches one of its local nodes, the internal forces of the bonds it owns are evaluated at the configuration current at that node and deposited into a backward and a forward impulse register, weighted respectively by the local interval preceding and the local interval following that node. The backward weight is omitted at a potential's first local node and the forward weight at its last. Each deposit places the contribution on both points joined by the bond, with the same signs and volume weights as the internal force accumulation.

Each advance from one global node to the next consists of three operations in this order. Velocities receive half of the forward impulses accumulated at the node being left, together with half of the external body-force impulse over the interval. Positions then advance using the resulting velocity. At the node just reached, both impulse registers are cleared, the active potentials update their bond states and deposit at the new configuration, and velocities then receive half of the backward impulses accumulated there together with the remaining half of the body-force impulse. Consequently the two half updates of an advance use forces evaluated at the two different configurations that bound it, and the first and last global nodes each contribute one half update rather than two.

Bond states are updated immediately before each deposit, on the bonds owned by the potential being updated. Bond states begin from initial_bond_states, and the body begins at rest with zero displacement.

Only the displacement field at the final time is returned, with one row per material point in the lattice ordering.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2) with B >= 1; any bond index lies outside the range of lattice; micromodulus does not have shape (N,) or is not finite and positive; critical_stretch_squared does not have shape (B,) or is not finite and positive; no_fail_points does not have shape (N,) or contains a value other than 0 or 1; body_force does not have shape (N, 2) or contains non-finite values; local_steps does not have shape (N,) or is not finite and positive; final_time is not finite and strictly positive; density is not finite and strictly positive; influence_exponent is not finite or is greater than or equal to 3; initial_bond_states does not have shape (B,) matching bonds, or contains a value other than 0 or 1; any local step is not a positive integer multiple of the smallest local step within a relative tolerance of 1e-12; final_time is not a positive integer multiple of the smallest local step, or of every local step, within a relative tolerance of 1e-12; any bond has non-positive reference length; any bond has non-positive deformed length.

```python
import numpy as np


def integrate_avv(lattice: np.ndarray, bonds: np.ndarray, micromodulus: np.ndarray,
                  critical_stretch_squared: np.ndarray, no_fail_points: np.ndarray,
                  body_force: np.ndarray, local_steps: np.ndarray, final_time: float,
                  density: float, influence_exponent: float,
                  initial_bond_states: np.ndarray) -> np.ndarray:
    '''Integrate the body to final_time with the asynchronous velocity-Verlet scheme.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    critical_stretch_squared : np.ndarray
        (B,) float array of squared critical stretch, one per directed bond.
    no_fail_points : np.ndarray
        (N,) array marking points exempt from failure, 1 for exempt and 0 otherwise.
    body_force : np.ndarray
        (N, 2) float array of external body-force density, constant in time.
    local_steps : np.ndarray
        (N,) float array giving each potential's local update interval.
    final_time : float
        Time at which the integration stops.
    density : float
        Mass density of the material.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.
    initial_bond_states : np.ndarray
        (B,) array of starting bond states, 1 for intact and 0 for broken.

    Returns
    -------
    displacement : np.ndarray
        (N, 2) float array of displacements at final_time, in the lattice ordering.
    '''
    displacement = np.zeros((len(np.asarray(lattice)), 2), dtype=float)
    return displacement  # placeholder
```

### Step 9

09_solve_plate_uy

Goal
----
Orchestrate the full pipeline and return the target scalar.



This step builds the material-point lattice, enumerates the directed bonds with the pre-crack removed, calibrates the micro-modulus for every horizon, computes each bond's squared critical stretch, estimates each region's critical time step, verifies the reference configuration, integrates the body to the final time, and reports the vertical displacement at the probe point.

Region identifiers are assigned from the point coordinates: a point belongs to the last region whose lower x-bound it exceeds, with region 0 as the default. The prescribed step of each region must be strictly below that region's critical time step.

Two properties of the reference configuration are checked before integrating. The internal force density of the undeformed, fully intact body must vanish identically, and no bond may fail at zero deformation. The bond states surviving the second check are the states from which the integration starts.

The external loading is a constant body-force density applied to the single outermost row of points along the top and bottom edges, positive on the top and negative on the bottom, of magnitude applied_stress divided by the local spacing. Points within no_fail_width of either edge are exempt from failure.

The probe point must coincide with a material point of the lattice, within an absolute tolerance of 1e-9 times the larger of height and one.

The returned value is the vertical displacement at the probe point at the final time, expressed in micrometres.

Raises ValueError if: region_steps does not have shape (R,) matching spacings, or is not finite and strictly positive; probe_point does not have shape (2,) or is not finite; no_fail_width is not finite or is negative; applied_stress is not finite; probe_point does not coincide with a material point; any region step is not strictly below that region's critical time step; the reference configuration carries non-zero internal force; any bond fails in the reference configuration. Conditions raised by the steps it calls propagate unchanged.

```python
import numpy as np


def solve_plate_uy(youngs_modulus: float, density: float, fracture_energy: float,
                   influence_exponent: float, x_bounds: np.ndarray, spacings: np.ndarray,
                   height: float, horizon_ratio: float, crack_y: float, crack_x_max: float,
                   no_fail_width: float, applied_stress: float, region_steps: np.ndarray,
                   final_time: float, probe_point: np.ndarray) -> float:
    '''Run the full pipeline and return the probe displacement in micrometres.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus E of the material.
    density : float
        Mass density.
    fracture_energy : float
        Critical energy release rate G_c.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.
    x_bounds : np.ndarray
        (R, 2) float array of region x-intervals, contiguous and increasing.
    spacings : np.ndarray
        (R,) float array of grid spacings, one per region.
    height : float
        Extent of the plate in y, spanning [0, height].
    horizon_ratio : float
        Ratio m in delta = m * spacing.
    crack_y : float
        Ordinate of the horizontal pre-crack line.
    crack_x_max : float
        Largest abscissa at which the pre-crack severs a bond.
    no_fail_width : float
        Width of the failure-exempt layer at the top and bottom edges.
    applied_stress : float
        Magnitude of the applied edge traction.
    region_steps : np.ndarray
        (R,) float array of local time steps, one per region.
    final_time : float
        Time at which the integration stops.
    probe_point : np.ndarray
        (2,) float array giving the reference coordinates of the probe point.

    Returns
    -------
    u_y : float
        Vertical displacement at the probe point at final_time, in micrometres.
    '''
    u_y = 0.0
    return u_y  # placeholder
```
