# Physics-Condensed_Matter_Physics-8

## Background

A periodic cell that carries more than one physical field couples them at their common boundaries, and the coupling operators are boundary integrals rather than volume integrals: one pairing the transverse motion of the plate with the liquid variable over the wetted face, another pairing the liquid variable with the elevation over the free surface. Neither introduces degrees of freedom of its own. The liquid itself contributes no stiffness to the coupled problem in the usual sense, because an incompressible irrotational liquid stores no energy in compression; its operator is the Dirichlet form of the potential, and the only elastic restoring force above the plate is the gravitational one that the displaced free surface feels. Because that liquid operator carries no inertia, the liquid variable can be removed from the eigenvalue problem exactly, at every wave vector, leaving equivalent inertia operators acting on the fields that remain.

Two features of the periodic bookkeeping decide whether the result is usable. The first is that the phase condition relates degrees of freedom on opposite faces of the cell, so each field has its own set of independent degrees of freedom and its own phase-carrying map onto the full set, and the reduction must produce Hermitian operators if the frequencies are to come out real. The second is that the liquid operator of a cell with no value of the potential imposed anywhere is not invertible at every wave vector, so the order in which the phase condition and the elimination of the liquid variable are applied is not free.

For the physics, the quantity that controls how strongly a mode feels the liquid is the product of the in-plane wavenumber and the layer depth. Where that product is large the liquid motion is confined to a boundary layer at the plate and the upper boundary is not felt at all, so the two descriptions must agree. Where it is small the whole layer moves with the plate, the two descriptions disagree most, and which of them is right is decided by whether the liquid displaced by a crest of the plate has anywhere to go.

## Problem

A two-dimensional phononic crystal plate opens flexural band gaps by periodically modulating its bending rigidity and its inertia, and those gaps are a property of the Bloch dispersion of the lattice rather than of any finite sample. Immersing the lattice changes that dispersion, and the change is not a renormalisation of the dry problem: a liquid layer resting on the plate has to be accelerated with it, and how much liquid moves depends both on the in-plane wavelength of the mode and on what closes the layer from above. Close the layer with a rigid lid and the liquid is trapped, so the inertia it adds per unit area grows without bound as the in-plane wavenumber falls. Leave the upper boundary free and the liquid can be displaced upward instead, the trapped-inertia growth is removed, and the free surface enters as a field of its own, an elevation that is periodic on the same lattice and that carries a gravitational restoring force. The immersed lattice therefore carries three coupled fields on one periodic cell, the flexural motion of the plate, a scalar potential whose gradient is the displacement of the irrotational liquid, and the elevation of the free surface, and the Bloch phase condition applies to all three across the cell boundary. Your task is to quantify what admitting the free surface, rather than closing the liquid with a rigid lid, does to a named band gap of the immersed lattice, and to establish how that effect decays as the layer deepens.

Build two descriptions of the same cell and compare them. In the hydro-elastic description the elevation is an independent field and the liquid couples the plate to it. In the added-mass description the elevation is suppressed, which closes the liquid with a rigid lid and leaves it acting on the plate as inertia alone. Solve both on the same mesh, at the same Bloch wave vectors, with the same liquid, and take the dry cell, carrying no liquid at all, as a third reference. Using a scalar potential for the liquid rather than its pressure is what keeps the assembled pencil symmetric, so choose the liquid variable accordingly.

Use the following deterministic configuration.

- Square lattice of side $a = 0.2$ m in both in-plane directions, one centred square inclusion per cell of side $0.12$ m, and a uniform plate thickness of $0.02$ m.
- The plate is moderately thick and carries only its flexural degrees of freedom, the transverse deflection and the two rotations of the transverse normal, with a shear correction factor of $5/6$. Each rotation is named by the plane it acts in rather than the axis it turns about: the first is the one that pairs with the slope of the deflection along the first in-plane axis in the transverse shear strain, the second with the slope along the second axis. In-plane stretching is outside the model.
- Matrix material: Young modulus $6 \times 10^{6}$ Pa, Poisson ratio $0.47$, density $1290$ kg m$^{-3}$. Inclusion material: Young modulus $411 \times 10^{9}$ Pa, Poisson ratio $0.28$, density $19300$ kg m$^{-3}$.
- The liquid is incompressible, inviscid and irrotational, of density $1000$ kg m$^{-3}$, in small-amplitude motion about hydrostatic rest under a gravitational acceleration of $9.8$ m s$^{-2}$. It fills the whole cell directly above the plate. Its lower boundary is the plate, its upper boundary is the free surface, and it is bounded laterally only by the lattice periodicity. Surface tension is neglected.
- The graded layer depth is $h = 0.01$ m and the reference layer depth is $h = 0.2$ m.
- Discretisation: the cell is divided into $10 \times 10$ equal square four-node elements in plane, and the plate and the free surface share that division. The liquid occupies the same $10 \times 10$ columns and is divided through its depth into $4$ layers of equal thickness by eight-node hexahedral elements, whose faces meet the plate below and the free surface above node for node.
- Integration: the bending energy of a plate element and its section inertia, both the translational and the rotary part, are integrated with the two-by-two Gauss rule, while its transverse shear energy is evaluated at the element centre alone. Every inertia and every interface operator is the consistent one; nothing anywhere is lumped onto its diagonal. Liquid volume integrals use the two-by-two-by-two rule and every interface integral the two-by-two rule on the face concerned.
- Bloch sampling: the zone-boundary segment running from $X = (\pi / a, 0)$ to $M = (\pi / a, \pi / a)$, divided into $32$ equal steps, giving $33$ wave vectors with both endpoints included. This segment is used in place of the whole boundary of the irreducible Brillouin zone because the assignment of a branch to a family stays determinate along it, whereas approaching the zone centre the lowest branch of each family falls towards zero frequency and the two hybridise, so that the assignment, and with it the numbering of the structural branches, stops being determinate there. Determinate does not mean separated in frequency, and the two must not be confused: at the graded depth the free-surface family also happens to lie wholly below the structural family at every sampled wave vector, but at the reference depth the two overlap in frequency, so a rule that splits the families by any frequency threshold will misclassify there. The split is the one prescribed under branch bookkeeping below, by count and by elevation participation, at both depths.
- Branch bookkeeping: at each wave vector the immersed cell returns one branch per independent degree of freedom of its reduced problem. The free-surface family accounts for exactly as many of those branches as the cell has independent elevation degrees of freedom. The branches that remain, taken in ascending frequency, are the structural branches and are numbered from one. Every branch of the added-mass cell and of the dry cell is structural.
- The signed width of the gap between structural branches five and six is the smallest value of the sixth branch over the $33$ wave vectors minus the largest value of the fifth branch over the same set. It is negative when the two branches overlap.
- Frequencies are ordinary frequencies in hertz. IEEE float64 arithmetic throughout.

State in your reasoning the number of independent elevation degrees of freedom of the cell and the number of branches the free-surface family accounts for; the frequency of the lowest free-surface branch at $M$, together with the value that the closed-form dispersion of surface gravity waves on a layer of the same depth gives at the same wavenumber, as an independent check on the free-surface implementation; the frequency of the highest free-surface branch and of the lowest structural branch at $M$, which show how far apart the two families sit at that corner; the signed gap width of the dry cell; the two signed gap widths at the graded depth; the two signed gap widths at the reference depth together with their difference; and which sampled wave vector carries each of the two edges of the hydro-elastic gap at the graded depth. Settle, rather than assume, three things the configuration prescribes without explaining: in what order the Bloch phase condition and the elimination of the liquid variable have to be applied, and why; what role the gravitational restoring force of the free surface plays in the value you report; and what justifies neglecting surface tension for the value you report, given that the capillary term is not negligible for every branch the free surface carries. Situate the computation against the published work it reproduces, reporting the following from that work rather than from your own computation: the two frequencies bounding the second band gap of its dry cell; the two frequencies bounding the first band gap of its rigid-lid cell at a layer depth of $0.01$ m, which that work describes as all but destroyed; the capillary length it estimates for water, and the length of the system it compares that against when it sets surface tension aside; what it reports happens to the band structure at a layer depth of $1$ m when the inertia coupling the plate to the free surface is scaled down; and which variable it adopts for the liquid in place of the liquid displacement field, together with the reason it gives for preferring it. These are properties of the published configuration and of the published method rather than of the configuration fixed above, and they are not expected to coincide with your own numbers. Report as the final answer the signed gap width of the hydro-elastic cell minus that of the added-mass cell, at the graded depth of $0.01$ m, in hertz, to at least six significant figures; it is graded to an absolute tolerance of $0.3$ Hz.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, including every quantity the reporting paragraph above asks for, without turning the response into a general pipeline summary. Do not paste the assembled operators, full branch tables, or per-wave-vector frequency listings.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_assemble_plate_bending_cell

Goal
----
The lattice is a flat plate of uniform thickness whose material alternates between a compliant matrix and a stiff, dense square inclusion repeated on a square lattice. Because the plate is flat and its section is symmetric about the mid-plane, stretching of the mid-plane and bending of the section do not couple, and only the bending family is driven by a liquid resting on the face. The cell therefore carries three generalised displacements at each node, the transverse deflection $w$ and the two rotations of the transverse normal, and the moderately thick description is used, in which those rotations are independent of the slope of $w$ so that transverse shear is a strain of its own rather than a constraint.

Two element integrals make up the cell. The bending energy pairs the gradients of the two rotations through the section-integrated plane-stress moduli, which carry a factor of the cube of the thickness. The transverse shear energy pairs the difference between the slope of the deflection and the rotation, through the shear modulus times the thickness times the shear correction factor. Integrating both with the same rule makes a four-node element far too stiff in bending, because the bilinear deflection cannot match the bilinear rotations closely enough to keep the shear strain small, so the shear term is evaluated at one point while the bending term keeps the two-by-two rule. That choice is a property of the element, not of the physics, and it moves every frequency the cell predicts.

The inertia of the section is likewise section-integrated: the deflection carries the density times the thickness, and each rotation carries the density times the cube of the thickness over twelve, the rotary inertia. The two phases differ in all three constants, and a node sitting on the phase boundary receives contributions from elements of both phases, which is what makes the assembled operators those of a genuine composite rather than of an averaged plate.

Nodes are numbered along the first in-plane axis fastest, so the node at in-plane indices $(i, j)$ is numbered $i + j (n + 1)$ for a division into $n$ elements per side, and the three degrees of freedom of a node occupy consecutive positions in the order deflection, then the rotation of the transverse normal in the plane spanned by the first in-plane axis and the thickness direction, then the rotation in the plane spanned by the second in-plane axis and the thickness direction. The first rotation is the one that pairs with the slope of the deflection along the first axis in the transverse shear strain, and the second with the slope along the second axis. Reading them instead as rotations about those axes exchanges the pair and transposes the cross-curvature terms. An element takes its four nodes anticlockwise starting from the corner of lowest indices. An element belongs to the inclusion when its centre lies inside the inclusion square, and the configuration is required to place the inclusion boundary on element boundaries so that no element is split.

```python
def assemble_plate_bending_cell(
    n_elem: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
) -> dict:
    """Assemble the unreduced bending stiffness and inertia of the two-phase plate cell.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    inclusion_side : float
        Side of the centred square inclusion in metres.
    plate_thickness : float
        Uniform plate thickness in metres.
    shear_factor : float
        Transverse shear correction factor.
    matrix_constants : np.ndarray
        Young modulus, Poisson ratio and density of the matrix phase, shape (3,).
    inclusion_constants : np.ndarray
        Young modulus, Poisson ratio and density of the inclusion phase, shape (3,).

    Returns
    -------
    dict
        Under the keys n_dof, n_inclusion_element, stiffness and mass.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when any length is not finite and above zero, when inclusion_side does not sit below cell_side, when the inclusion boundary does not fall on element boundaries, when shear_factor is outside the half-open interval from zero to one, when either constants array does not hold three finite entries, or when a modulus or density fails to sit above zero or a Poisson ratio falls outside the open interval from minus one to one half.
    """
    return
```

### Step 2

02_assemble_fluid_potential_stiffness

Goal
----
The liquid above the plate is incompressible, inviscid and irrotational, so its displacement is the gradient of a scalar potential and incompressibility makes that potential harmonic throughout the layer. The weak form of the harmonic condition is the Dirichlet form, the volume integral of the squared gradient of the potential, and nothing else: an incompressible liquid stores no energy in compression, so the operator this step assembles carries no material constant at all, not even the density. The density enters the coupled problem only where the liquid exchanges force with the plate or with the free surface, which is the next step.

The operator therefore has the dimensions of a length rather than of a stiffness, and the potential has the dimensions of an area, its gradient being a displacement. Reading those dimensions is the quickest check that the density has not been smuggled in here.

The layer is meshed with eight-node hexahedral elements whose in-plane division matches the plate exactly, so that the lowermost face of the liquid column shares its four nodes with the plate element beneath it and the uppermost face shares its four nodes with the free-surface element above it. That conformity is what allows the two couplings to be assembled without any interpolation between meshes. Within the layer the division through the depth is into equal slices.

Nodes are numbered with the first in-plane index fastest, then the second, then the layer index, so that the node at in-plane indices $(i, j)$ in layer $k$ is numbered $i + j (n + 1) + k (n + 1)^2$ for a division into $n$ elements per side. Layer zero is the face wetted by the plate and layer $n_z$ is the free surface. An element takes its four lower nodes anticlockwise from the corner of lowest indices, then the four upper nodes in the same order.

A property of the assembled operator matters later. Because no value of the potential is imposed anywhere on the cell, a potential that is constant throughout the liquid lies in its null space: a constant potential has zero gradient and therefore describes no motion. The operator is consequently singular before any periodic condition is applied, and whether it stays singular afterwards depends on the wave vector.

```python
def assemble_fluid_potential_stiffness(
    n_elem: int,
    cell_side: float,
    fluid_depth: float,
    n_layer: int,
) -> dict:
    """Assemble the Dirichlet form of the liquid displacement potential over the layer.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell in plane, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    fluid_depth : float
        Depth of the liquid layer in metres.
    n_layer : int
        Number of equal slices through the depth, one or more.

    Returns
    -------
    dict
        Under the keys n_dof, layer_thickness and stiffness.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when n_layer is not an integer of one or more, or when cell_side or fluid_depth is not finite and above zero.
    """
    return
```

### Step 3

03_assemble_interface_operators

Goal
----
Three boundary integrals close the coupled cell, and none of them introduces a degree of freedom of its own. The first lives on the wetted face of the plate and pairs the transverse deflection with the liquid potential of the nodes directly beneath the free surface, on the lowermost face of the liquid column. It carries the no-penetration condition in one direction and the pressure the liquid exerts on the plate in the other, and it is the same operator in both because the two conditions are adjoint. Only the deflection appears in it: the rotations of the transverse normal do no work against a pressure acting along the normal, so every row belonging to a rotation is empty.

The second lives on the free surface and pairs the elevation with the liquid potential on the uppermost face of the column. It states that the normal displacement of the liquid at the surface is the elevation, which is what keeps the liquid incompressible while the surface rises and falls.

The third is the only elastic operator above the plate. Displacing the free surface by an elevation raises liquid above the undisturbed level and leaves a deficit below it, and gravity supplies the restoring pressure, proportional to the liquid density, the gravitational acceleration and the elevation. Its weak form is that product times the surface integral of the elevation against itself, so this operator alone carries the density and the gravitational acceleration.

All three are built from the same object, the surface integral of the bilinear shape functions against one another over a square face, because the plate, the liquid face and the free surface share the same in-plane division and the same interpolation. Evaluating that integral with the two-by-two rule gives the consistent form, in which a node is coupled to its neighbours. Replacing it with its row sums, the lumped form, is a different and cruder operator and changes every coupled frequency.

The signs of the two coupling operators can be taken either way provided each is used consistently, because reversing the sense of the interface normal reverses both the force the liquid applies to the plate and the displacement the plate imposes on the liquid. The eigenvalues of the coupled cell do not see the choice.

Node numbering follows the two preceding steps: the free surface is numbered like the plate, with the first in-plane index fastest, and the liquid nodes carry the layer index slowest, so layer zero is the wetted face and layer $n_z$ is the free surface.

```python
def assemble_interface_operators(
    n_elem: int,
    cell_side: float,
    n_layer: int,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Assemble the two interface couplings and the gravitational restoring operator.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell in plane, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    n_layer : int
        Number of slices through the depth of the liquid, one or more.
    fluid_density : float
        Liquid density in kilogram per cubic metre.
    gravity : float
        Gravitational acceleration in metre per second squared.

    Returns
    -------
    dict
        Under the keys n_surface_dof, fsi_coupling, surface_coupling and surface_stiffness.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when n_layer is not an integer of one or more, or when cell_side, fluid_density or gravity is not finite and above zero.
    """
    return
```

### Step 4

04_reduce_cell_by_bloch_phase

Goal
----
Bloch's theorem says that a field on a lattice-periodic medium repeats from cell to cell up to a phase fixed by the wave vector and the lattice vector. On a single cell that is a linear constraint: the degrees of freedom on the face at the far end of a lattice vector equal those on the near face multiplied by the phase, and the degrees of freedom at the far corner pick up the product of the two phases. Each of the three fields carries the constraint separately, because each has its own set of boundary degrees of freedom, but all three carry the same two phases, since all three are periodic on the same lattice. The phase is taken with the positive exponent throughout: a field $u$ satisfies $u(x + a) = exp(i k . a) u(x)$ for a lattice vector $a$, so the face at the far end of the first lattice vector carries $exp(i k_x d)$ times the near face, the face at the far end of the second carries $exp(i k_y d)$, and the far corner carries the product of the two, where $d$ is the cell side. The opposite convention conjugates every reduced operator and leaves every eigenvalue of the cell unchanged, so it is fixed here only to make the returned arrays comparable entry by entry.

The constraint is a map from the independent degrees of freedom to the full set. With the in-plane indices running from zero to $n$ inclusive, the independent set is the nodes with both indices below $n$, and a node with an index equal to $n$ is the image of the node at index zero along that axis, carrying the corresponding phase. The liquid keeps its full set of layers, since the depth direction is not periodic. Each field therefore has $n^2$ independent nodes in plane, the plate three degrees of freedom on each of them, the free surface one, and the liquid one on each of $n_z + 1$ layers.

Every operator is then congruence-transformed by that map. The transform to use is the conjugate transpose on the left, not the plain transpose: the variational statement pairs a field with the conjugate of a test field drawn from the same space, and only the conjugate pairing returns Hermitian operators and therefore real squared frequencies. The plain transpose returns complex symmetric operators whose eigenvalues are complex, which is not a discretisation error but a different and wrong problem.

Because the map has exactly one non-zero entry in each row, the congruence is a scatter rather than a dense product: an entry of the full operator is multiplied by the conjugate of the phase of its row and the phase of its column, and accumulated into the independent pair that its row and column belong to. Both the square operators and the two rectangular couplings transform this way, the couplings with the map of their row field on the left and the map of their column field on the right.

```python
def reduce_cell_by_bloch_phase(
    plate_stiffness: np.ndarray,
    plate_mass: np.ndarray,
    fluid_stiffness: np.ndarray,
    fsi_coupling: np.ndarray,
    surface_coupling: np.ndarray,
    surface_stiffness: np.ndarray,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    wave_vector: np.ndarray,
) -> dict:
    """Impose the Bloch phase condition on all three fields and reduce every operator.

    Parameters
    ----------
    plate_stiffness : np.ndarray
        Unreduced plate bending stiffness.
    plate_mass : np.ndarray
        Unreduced plate inertia.
    fluid_stiffness : np.ndarray
        Unreduced Dirichlet form of the liquid potential.
    fsi_coupling : np.ndarray
        Unreduced plate to liquid coupling.
    surface_coupling : np.ndarray
        Unreduced free surface to liquid coupling.
    surface_stiffness : np.ndarray
        Unreduced gravitational restoring operator of the free surface.
    n_elem : int
        Elements per side of the square cell in plane, two or more.
    n_layer : int
        Number of slices through the depth of the liquid, one or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    wave_vector : np.ndarray
        The two in-plane components of the Bloch wave vector in radian per metre, shape (2,).

    Returns
    -------
    dict
        Under the keys plate_stiffness, plate_mass, fluid_stiffness, fsi_coupling, surface_coupling and surface_stiffness.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when n_layer is not an integer of one or more, when cell_side is not finite and above zero, when wave_vector does not hold two finite entries, or when any operator does not have the shape its field and the counts imply.
    """
    return
```

### Step 5

05_condense_potential_to_mass_operators

Goal
----
The liquid contributes no inertia term of its own to the coupled cell: its operator is the Dirichlet form of the potential, and the potential appears in the equations of the plate and of the free surface only through the two interface couplings and through second time derivatives. The row of the coupled system belonging to the potential is therefore algebraic rather than dynamic, and the potential can be eliminated exactly, at every wave vector, without approximation and without introducing any frequency dependence.

Solving that algebraic row for the potential in terms of the plate motion and the elevation, and substituting the result back, leaves three equivalent inertia operators, each of them the liquid density times one coupling times the inverse of the liquid operator times the conjugate transpose of a coupling. Taking the plate coupling on both sides gives the added mass, the inertia the liquid lends to the plate. Taking the surface coupling on both sides gives the surface mass, the inertia the liquid lends to the motion of the free surface. Taking one of each gives the coupling mass, which is not an inertia of anything on its own: it is the operator through which an acceleration of the plate drives the free surface and an acceleration of the free surface reacts on the plate. Set it to zero and the two fields separate completely, the plate is left with the added mass alone, and the cell returns exactly the rigid-lid problem.

The elimination has to be performed on operators that already carry the Bloch phase. The constraint that ties opposite faces of the cell is a constraint on the liquid as much as on the plate, so imposing it after the potential has been removed applies it to the wrong variables and gives a different, wrong answer. There is also a plain arithmetic reason: the liquid operator of the unconstrained cell is singular, because a potential constant throughout the liquid describes no motion, so its inverse does not exist at all until the reduction has removed that freedom. The reduction removes it for every wave vector except those at which the phases are all unity, and at those the operator is still singular and the elimination is impossible.

Because the liquid operator is Hermitian and, away from those wave vectors, positive definite, a Cholesky factorisation performs the two solves. It does not on its own detect the exceptional case: assembly round-off lifts the null mode enough that the factorisation of the singular operator succeeds and returns a meaningless inverse, so the factor has to be inspected as well. The ratio of the smallest to the largest entry on the diagonal of the factor is the usable measure. On the wave vectors of a zone-boundary locus it is of order one third; at a wave vector where the phases are all unity it collapses to a few parts in ten million, and everything computed from the inverse there is round-off.

```python
def condense_potential_to_mass_operators(
    fluid_stiffness: np.ndarray,
    fsi_coupling: np.ndarray,
    surface_coupling: np.ndarray,
    fluid_density: float,
) -> dict:
    """Eliminate the liquid potential and return the three equivalent inertia operators.

    Parameters
    ----------
    fluid_stiffness : np.ndarray
        Bloch-reduced Dirichlet form of the liquid potential, Hermitian of shape (m, m).
    fsi_coupling : np.ndarray
        Bloch-reduced plate to liquid coupling of shape (p, m).
    surface_coupling : np.ndarray
        Bloch-reduced free surface to liquid coupling of shape (s, m).
    fluid_density : float
        Liquid density in kilogram per cubic metre.

    Returns
    -------
    dict
        Under the keys added_mass, coupling_mass and surface_mass.

    Raises
    ------
    ValueError
        When fluid_stiffness is not square, when either coupling does not have as many columns as fluid_stiffness has rows, when any entry is not finite, when fluid_density is not finite and above zero, or when fluid_stiffness fails to be Hermitian positive definite, which is what happens at a wave vector where the phases are all unity.
    """
    return
```

### Step 6

06_solve_and_index_branches

Goal
----
With the liquid removed, the cell at one wave vector is a Hermitian pencil in two blocks. The plate block carries its own bending stiffness against its own inertia augmented by the added mass. The free-surface block carries the gravitational restoring operator against the surface mass. The two off-diagonal blocks of the inertia carry the coupling operator and its conjugate transpose, and there is no off-diagonal stiffness at all, because the plate and the free surface exchange no elastic force directly, only inertia through the liquid between them. Scaling both off-diagonal blocks by a common factor turns the coupling on and off continuously; at zero the pencil splits into two independent problems and the plate block is exactly the rigid-lid problem.

Three descriptions are wanted from the same operators. The dry cell is the plate stiffness against the plate inertia alone. The added-mass cell is the plate stiffness against the plate inertia plus the added mass, which is what suppressing the elevation leaves behind. The hydro-elastic cell is the full two-block pencil. All three are Hermitian with positive definite inertia away from the exceptional wave vectors, so the squared angular frequencies are real and non-negative and the ordinary frequency is their square root over two pi.

Sorting the branches of the two-block pencil by frequency mixes two families that mean different things, and the band structure of the plate is carried by only one of them. The free-surface family accounts for exactly as many branches of the spectrum as the cell has independent elevation degrees of freedom, which is what makes the two separable by counting: form for each branch the share of its inertia carried by the elevation block against the share carried by the plate block, using the two diagonal inertia blocks, which are the only positive semi-definite pieces available, and assign the family membership by rank on that share rather than by a threshold on it. Ranking is what guarantees the count, and the count is what keeps the numbering of the structural branches stable through the avoided crossings where a plate branch and a surface branch exchange character and neither share is decisively above one half.

```python
def solve_and_index_branches(
    plate_stiffness: np.ndarray,
    plate_mass: np.ndarray,
    added_mass: np.ndarray,
    coupling_mass: np.ndarray,
    surface_mass: np.ndarray,
    surface_stiffness: np.ndarray,
    model: str,
    coupling_factor: float,
) -> dict:
    """Solve the cell at one wave vector and split its branches into the two families.

    Parameters
    ----------
    plate_stiffness : np.ndarray
        Bloch-reduced plate bending stiffness of shape (p, p).
    plate_mass : np.ndarray
        Bloch-reduced plate inertia of shape (p, p).
    added_mass : np.ndarray
        Inertia the liquid lends to the plate, of shape (p, p).
    coupling_mass : np.ndarray
        Operator carrying the plate to free surface interaction, of shape (p, s).
    surface_mass : np.ndarray
        Inertia the liquid lends to the free surface, of shape (s, s).
    surface_stiffness : np.ndarray
        Bloch-reduced gravitational restoring operator, of shape (s, s).
    model : str
        One of dry, added_mass or hydro_elastic.
    coupling_factor : float
        Multiplier on both off-diagonal inertia blocks of the hydro-elastic pencil.

    Returns
    -------
    dict
        Under the keys n_surface_branch, structural_frequencies and surface_frequencies.

    Raises
    ------
    ValueError
        When model is not one of the three names, when coupling_factor is not finite and not negative, when any operator holds an entry that is not finite, when the shapes are mutually inconsistent, or when the assembled inertia fails to be positive definite so that the pencil has no real spectrum.
    """
    return
```

### Step 7

07_locate_gap_edges

Goal
----
A band gap is a property of a dispersion sampled over a locus of wave vectors, not of any one wave vector, so once the branches have been collected into a table with one row per sampled wave vector and one column per branch, the gap between two consecutive branches is read from the extremes of two columns. The upper edge of the lower branch is the largest value that column takes over the locus and the lower edge of the upper branch is the smallest value the next column takes, and the signed width is the second minus the first. Signed, because the two branches may overlap: a positive width is a frequency interval in which no branch of the sampled set propagates, and a negative width records by how much the branches cross in frequency without there being any wave vector at which both are present.

Reading a maximum and a minimum off a sampled locus is only meaningful if the extremes are resolved, and the cheap indicator is where they fall. If either sits strictly inside the locus, the width plainly depends on how finely the locus was sampled. If both sit at an endpoint, and the endpoints are the high-symmetry points that any sampling of the locus must contain, then the width did not move under the samplings tried; that is evidence and not proof, since a finer sampling can still find an interior extremum that a coarser one stepped over. This step therefore reports the positions of both extremes alongside the width, and a flag recording whether both fell on endpoints, and leaves the strength of that evidence to the caller.

One more reading comes free from the same table and is worth having, because the branch pair a lattice is designed around is not always the one an implementation reaches first. Scanning every consecutive pair and keeping the widest signed gap says which pair carries the largest frequency interval free of any sampled branch, and how wide it is. On a lattice whose gap has been closed by immersion that widest value is negative, and the pair that carries it is the least overlapped rather than the most open.

The table is required to be ordered: at each wave vector the branches are listed in ascending frequency, which is what makes a column a branch rather than an arbitrary selection. A row that is not ascending means the branches were collected without sorting, or that two families were mixed, and the width read from such a table means nothing, so it is rejected rather than reported.

```python
def locate_gap_edges(branch_table: np.ndarray, lower_branch: int) -> dict:
    """Read the signed gap width between two consecutive branches and locate its edges.

    Parameters
    ----------
    branch_table : np.ndarray
        Branch frequencies in hertz, shape (n_point, n_branch), each row ascending.
    lower_branch : int
        One-based index of the lower of the two branches bounding the gap.

    Returns
    -------
    dict
        Under the keys signed_width, lower_branch_top, upper_branch_bottom, lower_extreme_row, upper_extreme_row, edges_on_endpoints, widest_pair_lower_branch and widest_pair_width.

    Raises
    ------
    ValueError
        When branch_table is not a two-dimensional array of finite values with at least two rows and two columns, when any row is not in ascending order, or when lower_branch is not an integer of one or more that leaves a column above it.
    """
    return
```

### Step 8

08_report_sloshing_bandgap_contribution

Goal
----
This is the final step and it runs the whole chain from the top-level configuration, twice, once at the graded liquid depth and once at the reference depth, on the same lattice and with the same liquid. Step one assembles the bending stiffness and the section inertia of the two-phase moderately thick plate cell, with the transverse shear energy integrated at the element centre. Step two assembles the Dirichlet form of the liquid displacement potential over the layer, which carries no material constant at all. Step three assembles the two interface couplings and the gravitational restoring operator of the free surface, all three from the same consistent surface integral of the bilinear shape functions. None of those depends on the wave vector, so the plate and the interface operators are built once and the liquid operator once per depth. Step four then imposes the Bloch phase condition separately on the plate, on the liquid and on the free surface, and reduces every operator by the conjugated congruence at one wave vector. Step five eliminates the liquid potential from those already-reduced operators and returns the added mass, the coupling mass and the surface mass. Step six solves the dry, the added-mass and the hydro-elastic cells from the same operators and splits the hydro-elastic branches into the free-surface family and the structural family. Steps four to six repeat at every wave vector of the locus, filling one branch table per description, and step seven reads the signed gap width and the positions of its edges off each table.

The locus is the zone-boundary segment of the square lattice, running from the edge centre, whose components are pi over the lattice constant along the first in-plane axis and zero along the second, to the corner, whose components are pi over the lattice constant along both, divided into equal steps with both endpoints retained. Every wave vector on it has an in-plane magnitude of at least pi over the lattice constant, which keeps the assignment of a branch to a family determinate and the branch numbering meaningful. Determinate is not the same as separated in frequency: at the graded depth the free-surface family also lies wholly below the structural family, while at the reference depth the two overlap, so the split is made by elevation participation and by the count of elevation freedoms and never by a frequency threshold, which would misclassify at the deeper layer. Approaching the zone centre the lowest branch of each family falls towards zero and the two hybridise, so that the assignment of a branch to a family, and with it the numbering of the structural branches, stops being determinate there.

Only the depth changes between the two runs. The dry cell is therefore unaffected by it and must return the same width in both, which is a free consistency check on the chain and is enforced rather than merely reported. The graded number is the difference between the hydro-elastic and the added-mass widths at the graded depth. The same difference at the reference depth is what establishes the regime: at ten plate thicknesses the two descriptions of the liquid have converged and the difference has collapsed, while at half a plate thickness it is of the order of the whole dry gap.

```python
def report_sloshing_bandgap_contribution(
    graded_depth: float,
    reference_depth: float,
    lower_branch: int,
    n_step: int,
    n_elem: int,
    n_layer: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Run the chain at both depths and report the contribution the free surface makes to the gap.

    Parameters
    ----------
    graded_depth : float
        Depth of the liquid layer at which the answer is reported, in metres.
    reference_depth : float
        Deeper layer at which the same difference is evaluated, in metres.
    lower_branch : int
        One-based index of the lower of the two structural branches bounding the gap.
    n_step : int
        Steps into which the zone-boundary segment is divided.
    n_elem : int
        Elements per side of the square cell in plane.
    n_layer : int
        Slices through the depth of the liquid.
    cell_side : float
        Lattice constant of the square cell in metres.
    inclusion_side : float
        Side of the centred square inclusion in metres.
    plate_thickness : float
        Uniform plate thickness in metres.
    shear_factor : float
        Transverse shear correction factor.
    matrix_constants : np.ndarray
        Young modulus, Poisson ratio and density of the matrix phase, shape (3,).
    inclusion_constants : np.ndarray
        The same three constants for the inclusion phase, shape (3,).
    fluid_density : float
        Liquid density in kilogram per cubic metre.
    gravity : float
        Gravitational acceleration in metre per second squared.

    Returns
    -------
    dict
        Under the keys n_wave_vector, n_surface_branch, edges_on_endpoints, corner_surface_low, corner_surface_high, corner_structural_low, dry_gap, graded_hydro_gap, graded_added_mass_gap, reference_hydro_gap, reference_added_mass_gap, reference_contribution and sloshing_contribution.

    Raises
    ------
    ValueError
        When either depth is not finite and above zero, when reference_depth does not exceed graded_depth, when lower_branch or n_step is not an integer of one or more, when any configuration value rejected by an earlier step is passed through, when the cell returns fewer structural branches than the pair needs, or when the dry cell does not return the same gap width at the two depths.
    """
    return
```
