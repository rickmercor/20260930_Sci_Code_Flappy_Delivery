# Numerical dissipation of an implicit compact-support particle-grid march

## Background

The material point method carries a continuum on Lagrangian points and solves the momentum balance on a background grid, so every step passes the state through a particle-grid kernel twice. That kernel is where the method's character is decided. A kernel with a wide support is smooth, and smoothness is what removes the noise a point makes when it crosses a cell boundary, but a wide support also reaches into material the point should not feel: it smears interfaces, it opens artificial gaps at contacts, and it drags a larger stencil through every transfer. A kernel with a narrow support keeps the transfer local and cheap, but the piecewise linear kernel that first comes to mind has a discontinuous derivative and is exactly the source of the cell-crossing noise. Whether a kernel can be narrow and smooth at once, and whether such a kernel survives being embedded in an implicit solver, is the question this line of work asks.

Narrowing the support has a price that is easy to miss. Interpolation on a grid is first-order accurate only if the weights sum to one and the weighted node positions reproduce the point's own position; the second of those is what a kernel of unit support radius gives up, and giving it up costs angular momentum and introduces a systematic error in every velocity gradient. Recovering it without widening the support means changing the grid rather than the kernel, which is where a staggered family of lattices enters.

Implicit time integration changes what matters. The nodal velocities at the end of a backward Euler step are the stationary point of an incremental potential that balances the kinetic cost of departing from the transferred velocity against the strain energy the resulting deformation would store, and finding that point is a nonlinear minimization whose tangent inherits the sparsity of the kernel. Backward Euler is unconditionally stable, which is why it is chosen when the material is stiff, or the step is large, but stability is not accuracy: the scheme damps, and it damps more the larger the step. The transfers damp too, and for an entirely different reason, since each cycle projects the nodal field onto what the particle representation can carry and discards the remainder.

The two mechanisms are hard to tell apart from a single simulation because they act together once per step, and they respond to a refinement of the step in opposite ways. That makes the total energy a free body loses over a fixed span of physical time an informative diagnostic: the continuum problem conserves it exactly, so whatever is missing was removed by the discretization, and how the loss responds to the step size says which part of the discretization removed it.

## Problem

The particle-grid kernel of a material point method fixes the trade-off between smoothness and transfer locality, and one recent kernel design holds the support radius down to a single grid spacing while remaining twice continuously differentiable everywhere; it is deployed not on a single lattice but on the family of grids its own design prescribes, and until now it has been exercised only with explicit time integration. I have carried it into an implicit setting on a testbed of my own and want the numerical dissipation of the resulting scheme measured on the set-up below.

The testbed is a plane-strain square block of a compressible fixed corotated hyperelastic solid, ten cells on a side at a grid spacing of $0.02$ m, its lower left corner sitting on a node of the unshifted grid. The block holds four material points per cell laid out on the two-by-two sub-lattice at the cell quarter and three-quarter points in each direction, each carrying a reference volume of one quarter of a cell and the mass that volume implies, and every point starts at rest with the isochoric deformation gradient that has $1.2$ and its reciprocal on the diagonal. The material has a density of $1200$ kg m$^{-3}$, a shear wave speed of $6$ m s$^{-1}$ and a plane-strain dilatational wave speed of $14$ m s$^{-1}$. Nothing acts on the block from outside at any instant - no gravity, no traction and no kinematic constraint - so the exact motion holds fixed the sum of the kinetic energy of the material points and the strain energy they store.

Transfers in both directions are the angular-momentum-conserving affine ones. The grid update is backward Euler, obtained at every step by driving every component of the residual of the incremental potential below $10^{-10}$ kg m s$^{-1}$ with Newton's method started from the transferred nodal velocities, after which the particle positions and deformation gradients are advected with the converged nodal field. Allow at most eight Newton updates per step, and treat a step that has not met the tolerance after them as a failure rather than advancing the state. March twenty steps of $1$ ms and tell me what the result implies about the time step I chose. Your final answer must be a single number: the fraction of the initial total mechanical energy that this twenty-step march has removed. Two further marches are a required part of the answer, reaching that same end time instead by forty steps of $0.5$ ms and by eighty steps of $0.25$ ms. The scalars your reasoning must carry are: the two Lamé constants of the material; the initial total mechanical energy of the block; the total mechanical energy left at the end of the twenty-step march and its split into a kinetic and a strain part; the fraction removed by each of the two refinement marches; whether the Newton solve met that residual tolerance at every step and what that settles about whether the solver itself is responsible for the loss; and what the trend across the three marches implies for the step I chose and for what in the scheme removed the energy.

Before those numbers, name the conventions your scheme rests on: the transfer kernel and the extent of its support, why the design calls for more than one grid and how the grids' contributions are combined, the form you took for the affine momentum transfer, and the strain energy density you differentiated. State these explicitly even where the short reasoning leaves everything else out, since I cannot read your numbers without knowing which conventions produced them.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Inside <reasoning>, give the conventions and the scalars named above, with enough intermediate calculation to justify the values you report.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Inside <reasoning>, give the conventions and the scalars named above, with enough intermediate calculation to justify the values you report.
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

evaluate_kernel_weights

Goal
----
Build the particle-grid stencil of a staggered grid family and return the kernel weights, their spatial gradients and the node offsets for every particle.

```python
import numpy as np 

def evaluate_kernel_weights(positions: "np.ndarray", spacing: float,
                            grid_offsets: tuple, kernel_name: str) -> tuple:
    """Evaluate the particle-grid kernel over a family of staggered grids.

    Grid ``k`` of the family carries nodes at the positions
    ``(index + grid_offsets[k]) * spacing`` in each coordinate direction, so an
    offset of 0 gives the unshifted lattice and an offset of 1/2 gives the
    lattice shifted by half a cell in both directions.

    Two one-dimensional kernel profiles are supported, both even in their
    argument ``d`` measured in units of the grid spacing.

    ``"compact"`` is the compact kernel of the compact-kernel material point
    method, ``K(d) = 1 - |d| + sin(2 pi |d|) / (2 pi)`` for ``|d| <= 1`` and
    zero outside, whose derivative is ``sign(d) (cos(2 pi d) - 1)``. Its
    support radius is one spacing, so two nodes per direction and four nodes
    per grid enter in two dimensions. Weights returned for this
    profile are clamped at zero: evaluated in floating point at ``|d| = 1``
    the expression above gives about ``-3.9e-17`` rather than exactly 0,
    because ``sin(2 pi)`` is not exactly representable, and a negative weight
    there would give a node a negative mass. Every returned weight is
    therefore non-negative.

    ``"quadratic"`` is the quadratic B-spline, ``K(d) = 3/4 - d^2`` for
    ``|d| < 1/2``, ``K(d) = (3/2 - |d|)^2 / 2`` for ``1/2 <= |d| <= 3/2`` and
    zero outside. Its support radius is three halves of a spacing, so three
    nodes per direction and nine nodes per grid enter.

    The two-dimensional weight is the product of the one-dimensional profiles
    over the two directions, and the returned weight gradient is its gradient
    with respect to the particle position. The stencil of grid ``k`` starts at the integer
    index ``floor(x / spacing - grid_offsets[k] - shift)`` in each direction,
    with ``shift`` equal to 0 for a support radius of one spacing and to 1/2
    for a support radius of three halves. Local stencil slots are numbered
    ``s = side * a + b``, where ``side`` is the number of nodes per direction,
    ``a`` counts nodes along the first coordinate and ``b`` along the second.

    Parameters
    ----------
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the current particle positions
        in metres.
    spacing : float
        Background grid spacing in metres (spacing > 0).
    grid_offsets : tuple
        Offsets of the grids of the family, in units of the spacing; every
        entry must satisfy 0 <= offset < 1 and there must be at least one.
    kernel_name : str
        Either "compact" or "quadratic".

    Returns
    -------
    node_indices : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots, 2) holding the
        lattice indices of the stencil nodes.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return node_indices, weights, weight_gradients, node_offsets  # placeholder
```

### Step 2

transfer_particles_to_grid

Goal
----
Collect the stencil into an active node table and carry mass and affine momentum from the material points onto every grid of the family.

```python
import numpy as np


def transfer_particles_to_grid(masses: "np.ndarray", velocities: "np.ndarray",
                               affine_states: "np.ndarray", node_indices: "np.ndarray",
                               weights: "np.ndarray", node_offsets: "np.ndarray") -> tuple:
    """Carry mass and affine momentum from the material points onto the grids.

    Every node that appears in any particle stencil is active, whatever its
    mass. The active nodes are labelled by the triple (grid, first lattice
    index, second lattice index) and are ordered lexicographically by that
    triple, which fixes the row order of every nodal array in the rest of the
    calculation.

    The affine moment matrix of a particle is the weighted second moment of
    its node offsets, averaged over the grids of the family with the factor
    one over the number of grids. The momentum a particle deposits on a node
    is its mass times the weight times its velocity plus the affine state
    contracted with the inverse moment matrix and with the node offset. The
    nodal velocity is the deposited momentum divided by the deposited mass,
    and is set to zero at a node whose deposited mass vanishes.

    Parameters
    ----------
    masses : "np.ndarray"
        Array of shape (n_particles,) holding the particle masses in kilograms
        per unit thickness; every entry must be > 0.
    velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    affine_states : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the affine velocity states
        of the particles, in m^2 s^-1.
    node_indices : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots, 2) holding the
        lattice indices of the stencil nodes.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Returns
    -------
    node_keys : "np.ndarray"
        Integer array of shape (n_nodes, 3) holding the labels of the active
        nodes in lexicographic order.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the row of node_keys it refers to.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness.
    node_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities in m s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if an affine moment matrix is singular.
    """
    return node_keys, node_slots, node_masses, node_velocities  # placeholder
```

### Step 3

interpolate_grid_velocity

Goal
----
Read a nodal velocity field back onto the material points as a velocity, a velocity gradient and an affine velocity state.

```python
import numpy as np


def interpolate_grid_velocity(node_velocities: "np.ndarray", node_slots: "np.ndarray",
                              weights: "np.ndarray", weight_gradients: "np.ndarray",
                              node_offsets: "np.ndarray") -> tuple:
    """Read a nodal velocity field back onto the material points.

    Every returned quantity is a sum over the whole stencil of the particle,
    that is over every grid of the family and every local slot, divided by the
    number of grids.

    The particle velocity is the weighted sum of the nodal velocities. The
    velocity gradient has the derivative direction as its second index, so its
    entry (a, b) is the sum of the nodal velocity component a times component b
    of the weight gradient. The affine velocity state has the node offset
    direction as its second index, so its entry (a, b) is the weighted sum of
    the nodal velocity component a times component b of the node offset.

    Parameters
    ----------
    node_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities in m s^-1.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the row of node_velocities it refers to.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Returns
    -------
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    velocity_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the particle velocity
        gradients in s^-1.
    affine_states : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the affine velocity states
        of the particles, in m^2 s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal velocity array.
    """
    return particle_velocities, velocity_gradients, affine_states  # placeholder
```

### Step 4

evaluate_corotated_response

Goal
----
Evaluate the fixed corotated strain energy density and the first Piola-Kirchhoff stress of every material point from its deformation gradient.

```python
import numpy as np


def evaluate_corotated_response(deformation_gradients: "np.ndarray", shear_modulus: float,
                                lame_first: float) -> tuple:
    """Evaluate the fixed corotated energy density and first Piola stress.

    With ``R`` the rotation of the polar decomposition of the deformation
    gradient ``F`` and ``J`` its determinant, the strain energy density is

        ``psi = mu * ||F - R||_F^2 + (lambda / 2) * (J - 1)^2``,

    where ``||.||_F`` is the Frobenius norm, ``mu`` is the shear modulus and
    ``lambda`` is the first Lame constant. The first Piola-Kirchhoff stress is
    the derivative of that density with respect to the deformation gradient.

    In two dimensions the rotation is fixed by the requirement that ``R`` be
    orthogonal with unit determinant and that ``R^T F`` be symmetric, which
    determines it uniquely whenever the determinant of the deformation
    gradient is strictly positive.

    Parameters
    ----------
    deformation_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients;
        every determinant must be > 0.
    shear_modulus : float
        Shear modulus in pascals (shear_modulus > 0).
    lame_first : float
        First Lame constant in pascals (lame_first >= 0).

    Returns
    -------
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy density of
        every particle, in joules per cubic metre.
    first_piola : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the first Piola-Kirchhoff
        stress of every particle, in pascals.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return energy_densities, first_piola  # placeholder
```

### Step 5

evaluate_corotated_tangent

Goal
----
Differentiate the fixed corotated first Piola-Kirchhoff stress with respect to the deformation gradient to obtain the fourth-order material tangent.

```python
import numpy as np


def evaluate_corotated_tangent(deformation_gradients: "np.ndarray", shear_modulus: float,
                               lame_first: float) -> "np.ndarray":
    """Build the fourth-order tangent of the fixed corotated stress.

    The returned array holds the derivative of the first Piola-Kirchhoff
    stress with respect to the deformation gradient, so that entry
    ``(p, a, b, c, d)`` is the derivative of the stress component ``(a, b)`` of
    particle ``p`` with respect to the deformation gradient component
    ``(c, d)`` of the same particle. The tangent is symmetric under exchange of
    the index pair ``(a, b)`` with the index pair ``(c, d)``, since it is the
    second derivative of the strain energy density.

    Parameters
    ----------
    deformation_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients;
        every determinant must be > 0.
    shear_modulus : float
        Shear modulus in pascals (shear_modulus > 0).
    lame_first : float
        First Lame constant in pascals (lame_first >= 0).

    Returns
    -------
    tangents : "np.ndarray"
        Array of shape (n_particles, 2, 2, 2, 2) holding the material tangent
        of every particle, in pascals.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return tangents  # placeholder
```

### Step 6

assemble_potential_gradient

Goal
----
Evaluate the incremental potential of the implicit step at a trial nodal velocity and assemble its gradient with respect to the nodal degrees of freedom.

```python
import numpy as np


def assemble_potential_gradient(trial_velocities: "np.ndarray", initial_velocities: "np.ndarray",
                                node_masses: "np.ndarray", node_slots: "np.ndarray",
                                weight_gradients: "np.ndarray", reference_gradients: "np.ndarray",
                                energy_densities: "np.ndarray", first_piola: "np.ndarray",
                                volumes: "np.ndarray", step: float) -> tuple:
    """Evaluate the incremental potential and its nodal gradient.

    The incremental potential at a trial nodal velocity field ``vhat`` is

        ``E = (1/2) sum_i m_i |vhat_i - v_i|^2 + n_grids * sum_p V_p psi_p``,

    where ``m_i`` are the nodal masses, ``v_i`` the velocities the transfer
    deposited, ``V_p`` the reference volumes and ``psi_p`` the strain energy
    densities of the trial deformation gradients. The factor equal to the
    number of grids compensates the fact that each grid of the family already
    carries the whole body mass, so that the nodal kinetic term and the strain
    term are weighted alike.

    The trial deformation gradient of a particle is ``(I + step * L_p) F_p``,
    with ``F_p`` its reference deformation gradient and ``L_p`` the velocity
    gradient the trial field induces at the particle. The returned residual is
    the gradient of the potential with respect to the trial nodal velocities.

    Parameters
    ----------
    trial_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the trial nodal velocities in
        m s^-1.
    initial_velocities : "np.ndarray"
        Array of shape (n_nodes, 2) holding the nodal velocities the transfer
        deposited, in m s^-1.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness; every entry must be >= 0.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the node row it refers to.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    reference_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy densities of
        the trial deformation gradients, in joules per cubic metre.
    first_piola : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the first Piola-Kirchhoff
        stresses of the trial deformation gradients, in pascals.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    potential : float
        Value of the incremental potential in joules per unit thickness, as a
        native Python float.
    residual : "np.ndarray"
        Array of shape (n_nodes, 2) holding the gradient of the potential with
        respect to the trial nodal velocities, in kilogram metre per second.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal arrays.
    """
    return potential, residual  # placeholder
```

### Step 7

assemble_potential_hessian

Goal
----
Assemble the exact tangent matrix of the incremental potential with respect to the nodal velocity degrees of freedom.

```python
import numpy as np


def assemble_potential_hessian(tangents: "np.ndarray", reference_gradients: "np.ndarray",
                               weight_gradients: "np.ndarray", node_slots: "np.ndarray",
                               node_masses: "np.ndarray", volumes: "np.ndarray",
                               step: float) -> "np.ndarray":
    """Assemble the exact tangent matrix of the incremental potential.

    The matrix returned is the second derivative of the incremental potential
    with respect to the trial nodal velocities, the potential being the one
    whose first derivative the residual assembly returns. Degrees of freedom
    are ordered node by node, so the two components of node ``i`` occupy rows
    and columns ``2 i`` and ``2 i + 1``.

    Parameters
    ----------
    tangents : "np.ndarray"
        Array of shape (n_particles, 2, 2, 2, 2) holding the derivative of the
        first Piola-Kirchhoff stress with respect to the deformation gradient,
        evaluated at the trial deformation gradients, in pascals.
    reference_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_slots : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots) holding, for
        every stencil entry, the node row it refers to.
    node_masses : "np.ndarray"
        Array of shape (n_nodes,) holding the nodal masses in kilograms per
        unit thickness; every entry must be >= 0.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    hessian : "np.ndarray"
        Array of shape (2 * n_nodes, 2 * n_nodes) holding the tangent matrix,
        in kilograms per unit thickness.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if a slot refers to a node outside the
        nodal arrays.
    """
    return hessian  # placeholder
```

### Step 8

solve_newton_direction

Goal
----
Solve the tangent system for the Newton increment of the nodal velocities, shifting the tangent onto the positive definite cone when it is not already there.

```python
import numpy as np


def solve_newton_direction(residual: "np.ndarray", hessian: "np.ndarray") -> "np.ndarray":
    """Solve the shifted tangent system for the Newton increment.

    The matrix is first replaced by its symmetric part. A Cholesky
    factorization of that symmetric part is attempted; if it fails, a multiple
    of the identity is added and the attempt is repeated. The multiple starts
    at one thousandth of the mean of the diagonal entries of the symmetric
    part, or, when that mean is not strictly positive, at one thousandth of the
    largest absolute entry of the symmetric part, and it is doubled on every
    further failure. The increment returned solves the first system that
    factorizes, with the negative of the residual as the right-hand side.

    Parameters
    ----------
    residual : "np.ndarray"
        Array of shape (n_nodes, 2) holding the gradient of the incremental
        potential with respect to the nodal velocities. Every entry must be
        finite.
    hessian : "np.ndarray"
        Array of shape (2 * n_nodes, 2 * n_nodes) holding the tangent matrix,
        with the two components of node i in rows and columns 2 i and 2 i + 1.
        Every entry must be finite.

    Returns
    -------
    direction : "np.ndarray"
        Array of shape (n_nodes, 2) holding the Newton increment of the nodal
        velocities, in m s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if no shift within sixty doublings makes
        the matrix factorizable.
    """
    return direction  #placeholder
```

### Step 9

advance_particle_state

Goal
----
Advect the material points and update their deformation gradients from the converged velocity field of the step.

```python
def advance_particle_state(positions: "np.ndarray", deformation_gradients: "np.ndarray",
                           particle_velocities: "np.ndarray", velocity_gradients: "np.ndarray",
                           step: float) -> tuple:
    """Advect the material points and update their deformation gradients.

    The deformation gradient of a particle is multiplied on the left by the
    identity plus the step times the velocity gradient, in that order, so that
    the update composes the incremental motion with the deformation already
    accumulated. The position is advanced by the step times the particle
    velocity.

    Parameters
    ----------
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions at the
        start of the step, in metres.
    deformation_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the start of the step.
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities at the
        end of the step, in m s^-1.
    velocity_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the particle velocity
        gradients at the end of the step, in s^-1, with the derivative
        direction as the second index.
    step : float
        Time step in seconds (step > 0).

    Returns
    -------
    updated_positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions at the
        end of the step, in metres.
    updated_gradients : "np.ndarray"
        Array of shape (n_particles, 2, 2) holding the deformation gradients at
        the end of the step.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if the update would leave a particle with
        a non-positive determinant.
    """
    return updated_positions, updated_gradients  # placeholder
```

### Step 10

measure_mechanical_energy

Goal
----
Reduce the particle state to the kinetic, strain and total mechanical energy of the body and to its angular momentum about the centre of mass.

```python
def measure_mechanical_energy(masses: "np.ndarray", particle_velocities: "np.ndarray",
                              volumes: "np.ndarray", energy_densities: "np.ndarray",
                              positions: "np.ndarray") -> tuple:
    """Reduce the particle state to energies and to an angular momentum.

    The kinetic energy is one half of the mass weighted sum of the squared
    particle speeds. The strain energy is the sum of the strain energy
    densities weighted by the reference volumes. The total is their sum. The
    angular momentum is taken about the centre of mass of the particles, and
    in two dimensions it is the scalar out-of-plane component, that is the sum
    over particles of the mass times the cross product of the position
    measured from the centre of mass with the velocity.

    Parameters
    ----------
    masses : "np.ndarray"
        Array of shape (n_particles,) holding the particle masses in kilograms
        per unit thickness; every entry must be > 0.
    particle_velocities : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle velocities in
        m s^-1.
    volumes : "np.ndarray"
        Array of shape (n_particles,) holding the reference particle volumes in
        cubic metres per unit thickness; every entry must be > 0.
    energy_densities : "np.ndarray"
        Array of shape (n_particles,) holding the strain energy densities in
        joules per cubic metre; every entry must be >= 0.
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the particle positions in
        metres.

    Returns
    -------
    kinetic_energy : float
        Kinetic energy in joules per unit thickness, as a native Python float.
    strain_energy : float
        Strain energy in joules per unit thickness, as a native Python float.
    total_energy : float
        Sum of the kinetic and the strain energy, as a native Python float.
    angular_momentum : float
        Out-of-plane angular momentum about the centre of mass, in kilogram
        square metre per second per unit thickness, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above or if the shapes
        are mutually inconsistent.
    """
    return kinetic_energy, strain_energy, total_energy, angular_momentum  # placeholder
```

### Step 11

run_ck_mpm_energy_loss

Goal
----
Chain the sub-problem functions 01-10 over the whole implicit march of the released pre-strained block and return the fraction of its mechanical energy the scheme has lost.

```python
def run_ck_mpm_energy_loss(spacing: float = 0.02, cells: int = 10,
                           particles_per_cell: int = 4, pre_stretch: float = 1.20,
                           density: float = 1200.0, shear_speed: float = 6.0,
                           dilatational_speed: float = 14.0, step: float = 1.0e-3,
                           n_steps: int = 20, tolerance: float = 1.0e-10,
                           max_iterations: int = 8, kernel_name: str = "compact",
                           grid_offsets: tuple = (0.0, 0.5)) -> float:
    """Run the implicit march of the released block and return the energy loss.

    The body is a square of side ``cells * spacing`` whose lower left corner
    sits on a node of the unshifted grid. It is filled with
    ``particles_per_cell`` material points per cell, laid out as a square
    sub-lattice at the cell fractions ``(j + 1/2) / side`` for
    ``j = 0, ..., side - 1`` in each direction, with ``side`` the square root
    of the number of particles per cell. Every particle carries the reference
    volume ``spacing^2 / particles_per_cell``, the mass that volume and the
    density give, zero velocity, a zero affine state and the isochoric
    deformation gradient with ``pre_stretch`` and its reciprocal on the
    diagonal.

    The Lame constants follow from the density and the two plane-strain wave
    speeds. Every step deposits the points onto the grids, drives the residual
    of the incremental potential below the tolerance by Newton iteration
    started from the deposited velocities and stopped after at most
    ``max_iterations`` updates, then reads the converged field back and
    advects. No external loading acts at any point.

    The returned number is one minus the ratio of the total mechanical energy
    after the march to the total mechanical energy before it.

    Parameters
    ----------
    spacing : float
        Background grid spacing in metres (spacing > 0).
    cells : int
        Number of cells along each side of the block (cells >= 1).
    particles_per_cell : int
        Number of material points per cell; must be a perfect square >= 1.
    pre_stretch : float
        Stretch carried by the first diagonal entry of the initial deformation
        gradient (pre_stretch > 0).
    density : float
        Mass density in kilograms per cubic metre (density > 0).
    shear_speed : float
        Shear wave speed in m s^-1 (shear_speed > 0).
    dilatational_speed : float
        Plane-strain dilatational wave speed in m s^-1; it must exceed the
        shear speed times the square root of two.
    step : float
        Time step in seconds (step > 0).
    n_steps : int
        Number of steps to march (n_steps >= 0).
    tolerance : float
        Largest absolute entry of the residual accepted as converged, in
        kilogram metre per second (tolerance > 0).
    max_iterations : int
        Largest number of Newton updates per step (max_iterations >= 1).
    kernel_name : str
        Either "compact" or "quadratic".
    grid_offsets : tuple
        Offsets of the grids of the family, in units of the spacing.

    Returns
    -------
    energy_loss : float
        Fraction of the initial total mechanical energy the march has removed,
        as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above or if the initial
        state carries no mechanical energy, or if a Newton solve fails to meet
        the requested residual tolerance within ``max_iterations`` updates.
    """
    return energy_loss  # placeholder
```
