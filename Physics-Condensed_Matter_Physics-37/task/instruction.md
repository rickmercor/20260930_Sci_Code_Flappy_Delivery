# Physics-Condensed_Matter_Physics-37

## Background

Bragg scattering needs the period to be comparable with the wavelength, which at low frequency demands a structure too large to be useful, whereas local resonance does not: if each cell holds an element that resonates on its own, the crystal acquires a band around that resonance in which waves do not propagate however long they are compared with the cell. Turning that picture into a number requires care about which contrast does what, since a resonator that is merely rigid is a different object from one that is heavy and stiff together and the two limits give different answers.

A body floating alone in an elastic medium carries six zero-energy rigid displacements, three translations and three rotations. A reduction that retains only some of them returns only some of the branches, so the set retained has to be stated before any number computed from it means anything. The two families also answer to different inertias, and at a general quasi-momentum they need not be independent of one another.

## Problem

A phononic crystal built from heavy, stiff resonators embedded in a soft elastic matrix can stop waves whose wavelength is many times the spacing of the resonators, which no crystal relying on Bragg scattering can do. The mechanism is local resonance: each resonator oscillates on the compliance of the matrix around it, and over a band of frequencies above those resonances the crystal supports no propagating wave. Your task is to compute, in hertz, how far the upper edge of that first bandgap sits, at the stated finite contrast and at the quasi-momentum where the leading-order theory attains that edge, from where the leading-order theory places it, for one prescribed three-dimensional crystal.

The crystal is a cubic lattice of period $L$ holding one resonator per cell. Outside it sits a soft isotropic background with Lame parameters $(lam, mu)$ and density $rho$, the Lame operator being $mu$ times the divergence of the sum of the displacement gradient and its transpose plus $lam$ times the gradient of the divergence; inside, the Lame parameters are $(lam, mu)$ divided by $delta$ and the density is $rho$ divided by $eps$, so $delta$ and $eps$ are the reciprocals of the stiffness and density contrasts. Both are small and the wave speed contrast $tau = (delta / eps)^{1/2}$ is of order one, so the resonator is heavy and stiff in the same proportion rather than merely rigid. Displacement and traction are continuous across the resonator surface, and Floquet-Bloch theory reduces the crystal to the single cell at a fixed quasi-momentum.

The leading-order result is the one in the joint limit of small contrast, and it is reported alongside the graded quantity. In that limit the resonator moves as a rigid body, and a rigid body in three dimensions carries six independent motions: three translations and three rotations. Build the reduction on all six. That reduction is the leading-order theory. The crystal is to be solved at its stated finite contrast as well, as the two-phase elastic body it is, discretised on the same grid with the same elements and with the resonator carrying its own Lame parameters and density, at the single quasi-momentum where the leading-order upper edge is attained.

What fixes the frequencies of the resulting six branches is part of the problem and is not given here. Each branch is a generalised stiffness reduced against the inertia the uniform rigid resonator presents to the six motions, expressed in the same basis as the stiffness.

The three branches built on the translations each descend to zero frequency as the quasi-momentum goes to zero, so together they fill an interval running from zero up to the largest frequency any of them attains. The three built on the rotations do not descend to zero. The first bandgap is the interval between the top of the first family and the bottom of the second; its leading-order upper edge is the smallest frequency any rotational branch attains. The quantity graded here is the correction the finite contrast makes to that upper edge: the fourth lowest Bloch frequency of the two-phase cell at the quasi-momentum attaining the leading-order upper edge, minus that leading-order edge. Every extremum over the zone is to be taken over the sixteen quasi-momenta listed below and over nothing else, so that the answer is a finite computation rather than a search.

Use the following deterministic configuration.

- cubic unit cell of edge $L = 0.02$ m, gridded into $24$ trilinear hexahedral elements along each edge, so the element edge is $h = L / 24$ and the cell closes periodically onto $24^{3}$ distinct nodes, the node at integer triple $(i, j, k)$ sitting at $(i, j, k) h$
- the resonator is an axis-aligned rectangular box spanning $12$, $9$ and $6$ whole elements along the first, second and third axes, its low corner at element index $(24 - span)$ integer-divided by two in each direction, so that every element is wholly inside it or wholly outside it
- soft background with $lam = 1.5 \times 10^{6}$ Pa, $mu = 5.0 \times 10^{5}$ Pa and $rho = 1200$ kg per cubic metre
- contrasts $delta = 1.0 \times 10^{-2}$ and $eps = 5.0 \times 10^{-3}$
- element integration by the two-point Gauss rule in each direction, eight points, at local coordinates plus and minus one over the square root of three, with the twenty-four degrees of freedom of an element ordered as corner index times three plus component, and the eight corners ordered by the sign pattern with the first axis varying slowest
- quasi-periodicity imposed as $u(x + L n) = \exp(i\, alpha \cdot n)\, u(x)$ for every integer triple $n$, the lattice translation being $L n$ because $x$ is in metres, with the quasi-momentum $alpha$ dimensionless and running over the Brillouin zone from $-\pi$ to $\pi$ in each direction
- every linear system solved to a relative residual of $10^{-12}$ or tighter, and accumulated so that the reported figures do not move when the linear algebra threading changes
- the quasi-momenta to examine, written as multiples of $\pi$ and in this order: $(1, 0, 0)$, $(0, 1, 0)$, $(0, 0, 1)$, $(1, 1, 0)$, $(1, 0, 1)$, $(0, 1, 1)$, $(1, 1, 1)$, $(1/2, 0, 0)$, $(0, 1/2, 0)$, $(0, 0, 1/2)$, $(1, 1/2, 0)$, $(1/2, 1, 0)$, $(1, 1, 1/2)$, $(3/4, 1/4, 1/8)$, $(2/3, 1/3, 1/6)$ and $(1/4, 1/4, 1/4)$
- at each quasi-momentum the branch frequencies are sorted ascending, so branch zero is the lowest
- the figures below are quoted to eight significant figures, so the arithmetic must carry at least double precision
- the finite-contrast problem discretised on the same grid with the same elements, the resonator elements carrying the Lame parameters divided by $delta$ and the density divided by $eps$, with a consistent mass matrix integrated by the same two-point rule
- the eight lowest eigenfrequencies of that finite-contrast problem at the attaining quasi-momentum, each converged to a relative residual below $10^{-8}$

Report that correction in hertz, with its sign, to eight significant figures. The answer is graded within $0.01$ Hz.

Report alongside it, briefly and each to at least eight significant figures unless it is an integer: the leading-order upper edge in hertz and the finite-contrast edge in hertz; the eight lowest finite-contrast frequencies at the attaining quasi-momentum, and which of them is the first that is not subwavelength; the largest frequency each of the six branches attains over the listed quasi-momenta, summed over the branches; the lower edge of the gap in hertz and its width in hertz; in the leading-order reduction, the number of elements that carry an equation, and, each counted among the nodes those elements touch, the number of nodes carrying prescribed data and the number of free nodes, as exact integers; the volume of the resonator in cubic metres and its mass in kilograms; the inertia matrix the frequencies were reduced against, with the units of its entries; the six eigenvalues of the small matrix the branch frequencies are built from, at the quasi-momentum attaining the upper edge, formed with unit Cartesian displacements and unit rotations of the resonator surface and no further normalisation of that basis, stating the units they carry and naming the point the rotations were taken about; the quasi-momentum in the list above at which the upper edge is attained, and the one at which the lower edge is attained; the two endpoints in hertz of the interval that the dilute limit brackets the lower edge within for a ball of the same volume as the resonator; and which of those two endpoints a dilute arrangement of equal balls opens its first bandgap just above.

Everything named in the two paragraphs above is part of the answer and belongs inside the reasoning, the diagnostics and comparisons as much as the quantities that fix the edge itself. State, in one or two sentences each: why the stiffness contrast $delta$ does not appear in the branch frequencies even though it is the parameter scaling the traction across the resonator surface; why the three translational branch frequencies vanish with the quasi-momentum while the rotational ones do not, and what that implies about where each edge must be sought; and whether the translational and rotational families mix, naming the quasi-momenta in the list at which they do not and saying what that implies for a reduction that keeps only the translations; and what the finite contrast introduces that the leading-order reduction leaves out, and the direction in which each such effect moves the six subwavelength frequencies.

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

01_resolve_contrast_scaling

Goal
----
Report the resonator parameters and both contrasts explicitly. Which of delta and eps appears where is the single most consequential piece of bookkeeping in this problem, and working out which of them sets the resonant frequencies is part of the task rather than something this stage settles.

Returns
-------
dict, holding lam_in, mu_in and rho_in for the resonator, tau for the wave speed contrast, and c_s and c_p for the background.

```python
def resolve_contrast_scaling(
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
) -> dict:
    """Turn the background parameters and the two contrasts into the resonator parameters and the regime numbers.

    Parameters
    ----------
    lam : float
        First Lame parameter of the soft background in pascal.
    mu : float
        Shear modulus of the soft background in pascal, above zero.
    rho : float
        Density of the soft background in kilogram per cubic metre, above zero.
    delta : float
        Reciprocal stiffness contrast, above zero and below one.
    eps : float
        Reciprocal density contrast, above zero and below one.

    Returns
    -------
    dict
        Under the keys lam_in, mu_in, rho_in, tau, c_s and c_p.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when mu or rho fails to be above zero, when three lam plus two mu fails to be above zero, or when delta or eps falls outside the open interval from zero to one.
    """
    return
```

### Step 2

02_partition_unit_cell

Goal
----
The reduction this task rests on replaces the transmission problem across the boundary of D by a problem posed only outside it, so only the elements outside D are ever assembled. That splits the nodes into three classes, and getting the split right is the whole of this stage.

- Nodes strictly inside D touch no exterior element at all. They carry no equation and must be dropped; leaving them in produces a singular system with empty rows.
- Nodes lying on the boundary surface of D are where the Dirichlet data will later be imposed. A node is on that surface when it lies within the closed box and at least one of its three indices equals the low or the high face index in that direction.
- Every remaining node touched by an exterior element is free.

Return the exterior element list, the two node sets and the volume of D, which is the product of the three spans times h cubed. Return also the three edge lengths of the resonator in metre, which fix its volume and its shape. That volume enters the resonant frequency directly, so it is worth forming here rather than inferring it later. The element list is a list of element index triples and not an element-to-node connectivity table: row e holds the three integers (i, j, k) that locate element e on the grid, so the array has exactly three columns, and the node indices of its eight corners are formed later from those triples and the corner offsets.

Returns
-------
dict, holding h, volume_D, the exterior element list elements, the node index arrays surface_nodes and free_nodes, the counts n_surface and n_free, and the float array sides giving the three edge lengths of the resonator in metre.

```python
def partition_unit_cell(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
) -> dict:
    """Grid the cell, place the box resonator and split the nodes into surface, free and discarded.

    Parameters
    ----------
    lattice_constant : float
        Edge of the cubic unit cell in metre, above zero.
    n_side : int
        Elements along each edge of the cell, at least four.
    spans : tuple
        Three integers giving the elements spanned by the resonator along each axis.

    Returns
    -------
    dict
        Under the keys h, volume_D, elements, surface_nodes, free_nodes, n_surface,
        n_free and sides. sides is the float array of shape (3,) holding the three edge
        lengths of the resonator in metre, so sides is spans times h and their product
        is volume_D.
        elements is an integer array of shape (n_exterior, 3) containing element
        index triples. surface_nodes and free_nodes are one-dimensional integer
        arrays of flat node indices, each sorted ascending. For the periodic
        node (i, j, k), the flat index is (i * n_side + j) * n_side + k,
        with each coordinate first reduced modulo n_side.

    Raises
    ------
    ValueError
        When lattice_constant is not finite and above zero, when n_side is not an integer of four or more, when spans does not hold exactly three integers, or when any span falls below one or exceeds n_side minus two.
    """
    return
```

### Step 3

03_hex_element_stiffness

Goal
----
Use trilinear shape functions on the eight corners and integrate with the two-point Gauss rule in each direction, eight points in all at the local coordinates plus and minus one over the square root of three. That rule is exact for this integrand on an undistorted cube, so the matrix returned is the exact element energy and not an approximation of it; a one-point rule would be rank deficient and admit hourglass modes.

Order the twenty-four degrees of freedom so that the three components of a corner are adjacent, corner index times three plus component. Order the corners themselves by the sign pattern with the first axis varying slowest, so that corner index is four times the first bit plus two times the second bit plus the third, each bit being zero at the low face and one at the high face. The ordering is a convention, but it has to be the same convention the assembly stage uses, and returning it explicitly is what lets the next stage be written without guessing.

Build the constitutive matrix in Voigt form with the normal block carrying lam off the diagonal and lam plus two mu on it, and the three shear entries carrying mu, then form the strain-displacement matrix at each Gauss point and accumulate B transpose C B times the Jacobian determinant. The Jacobian of a cube of edge h is h over two times the identity, so the determinant is h cubed over eight and the derivative conversion is a factor two over h.

Returns
-------
dict, holding the 24 by 24 element_stiffness, the 8 by 3 corner_signs and the element volume.

```python
def hex_element_stiffness(
    lam: float,
    mu: float,
    h: float,
) -> dict:
    """Form the twenty-four by twenty-four stiffness of one trilinear cubic element of isotropic background.

    Parameters
    ----------
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal, above zero.
    h : float
        Element edge in metre, above zero.

    Returns
    -------
    dict
        Under the keys element_stiffness, corner_signs and volume.
        element_stiffness has shape (24, 24). Despite its name, corner_signs
        is the integer array of binary corner offsets in {0, 1}, with shape
        (8, 3), ordered as (0,0,0), (0,0,1), (0,1,0), (0,1,1),
        (1,0,0), (1,0,1), (1,1,0), (1,1,1). Reference-element signs
        used in the shape functions are 2 * corner_signs - 1.
        volume is the element volume h**3.

    Raises
    ------
    ValueError
        When lam is not finite, when mu is not finite and above zero, or when h is not finite and above zero.
    """
    return
```

### Step 4

04_assemble_bloch_exterior

Goal
----
Assemble only over the exterior elements, and only the phases, nothing else: the material is the background everywhere outside the resonator, so no element carries the resonator parameters. Using the resonator moduli here is the single largest error available in this problem, and it scales the answer by the reciprocal square root of the stiffness contrast.

Having built the matrix over all n_side cubed nodes, extract the three blocks the condensation needs, indexing degrees of freedom as node index times three plus component, with the free and surface degree of freedom lists each sorted ascending. Return the free to free block, the free to surface block and the surface to surface block. Nodes strictly inside the resonator never appear.

Returns
-------
dict, holding the sparse blocks K_ff, K_fc and K_cc, and the index arrays free_dofs and surface_dofs.

```python
def assemble_bloch_exterior(
    elements: np.ndarray,
    surface_nodes: np.ndarray,
    free_nodes: np.ndarray,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    n_side: int,
    alpha: tuple,
) -> dict:
    """Assemble the quasi-periodic background stiffness outside the resonator and split it into blocks.

    Parameters
    ----------
    elements : np.ndarray
        Index triples of the exterior elements, shape (n_exterior, 3).
    surface_nodes : np.ndarray
        Flat indices of the resonator surface nodes, sorted ascending.
    free_nodes : np.ndarray
        Flat indices of the free nodes, sorted ascending.
    element_stiffness : np.ndarray
        Twenty-four by twenty-four element stiffness.
    corner_signs : np.ndarray
        Eight corner offsets of shape (8, 3).
    n_side : int
        Elements along each edge of the cell.
    alpha : tuple
        Three finite floats giving the quasi-momentum, not all zero.

    Returns
    -------
    dict
        Under the keys K_ff, K_fc, K_cc, free_dofs and surface_dofs.

    Raises
    ------
    ValueError
        When elements is not a two-dimensional integer array of three columns, when surface_nodes or free_nodes is not a non-empty one-dimensional integer array of indices inside the cell, when element_stiffness is not twenty-four by twenty-four, when corner_signs is not eight by three, when n_side is not an integer of four or more, when alpha does not hold exactly three finite floats, or when alpha is the zero vector.
    """
    return
```

### Step 5

05_capacity_schur_matrix

Goal
----
What the map has to be tested against is the second thing this stage needs. In the static limit the interior problem is a Lame system with a traction-free surface, whose solutions are the six rigid motions of the resonator, three translations and three rotations. All six are retained here. The source's own reduction keeps only the three constant fields; that restriction is not a consequence of the quasi-periodicity, since a centred rotation multiplied by a cutoff that vanishes before the cell faces is an admissible quasi-periodic field with zero strain and zero traction on the resonator surface, and dropping it is what this stage undoes. The matrix is therefore six by six, built by testing the map against all six motions:

$$Q[i, j] = - integral over the resonator surface of (map applied to g_i) . g_j.$$

Collect the six motions as the columns of a matrix E and form Q as E conjugate-transposed times S times E. Column c for c in zero to two is the constant field e_c, one in component c at every surface node and zero elsewhere. Column three plus a is the rotation about axis a through the point reference_point supplied to this stage: at a surface node of position x, it is the vector e_a cross (x - reference_point), taken component by component. The minus sign in the definition is already accounted for: the outward normal of the region outside the resonator points into the resonator on that surface, so the condensed form returns the positive quantity directly, and Q comes out as the elastic energy stored outside by the field that equals g_i on the surface.

The first three columns are dimensionless and the last three carry metre, so the six diagonal entries of Q do not share a unit and its six eigenvalues are not a spectrum. Converting them into frequencies is the next stage's problem.

Two properties follow and both must be checked, because they are the only independent confirmation available that the assembly and the phases are right. Q is Hermitian, and Q is positive definite. Reject a matrix whose departure from Hermitian symmetry is large relative to its own diagonal, entry by entry rather than against the largest entry alone, since the two blocks differ by six orders of magnitude and an absolute measure would not see an asymmetric rotational block; reject rather than symmetrising it away, since symmetrising unconditionally conceals exactly the assembly error the check exists to catch; then symmetrise the accepted matrix so that the eigenvalues the next stage takes are real by construction rather than by luck.

Solve with the conjugate gradient method preconditioned by the diagonal, which is available because K_ff is Hermitian positive definite, iterating until the residual falls to rel_tol times the norm of the right-hand side, and verifying that against a freshly evaluated residual with an order of magnitude of slack, since at a tolerance near the roundoff floor the two differ by a few per cent. Accumulate every inner product as a plain summation over the elementwise product rather than through a threaded dot product, so the result does not shift when the linear algebra library changes how many threads it uses. The iteration counts are part of what this stage returns and must be the counts actually taken, so a direct factorisation is not a conformant implementation of this step even though it would reach the same residual; the prompt's tolerance wording governs the pipeline, this docstring governs this step.

Returns
-------
dict, holding the 6 by 6 capacity, its eigenvalues, the hermitian_defect and the six iteration counts.

```python
def capacity_schur_matrix(
    K_ff: "scipy.sparse.spmatrix",
    K_fc: "scipy.sparse.spmatrix",
    K_cc: "scipy.sparse.spmatrix",
    surface_dofs: np.ndarray,
    n_side: int,
    h: float,
    reference_point: np.ndarray,
    rel_tol: float,
) -> dict:
    """Condense the exterior stiffness onto the resonator surface and test it against the six rigid motions of the resonator.

    Parameters
    ----------
    K_ff : scipy.sparse.spmatrix
        Free to free block, Hermitian positive definite.
    K_fc : scipy.sparse.spmatrix
        Free to surface block.
    K_cc : scipy.sparse.spmatrix
        Surface to surface block.
    surface_dofs : np.ndarray
        Degree of freedom indices of the surface block.
    n_side : int
        Elements along each edge of the cell, used to place the surface nodes.
    h : float
        Element edge in metre.
    reference_point : np.ndarray
        The point about which the rigid rotations are taken, in metre, shape (3,).
    rel_tol : float
        Relative residual at which the iteration stops.

    Returns
    -------
    dict
        Under the keys capacity, eigenvalues, hermitian_defect and iterations.
        hermitian_defect is the departure from Hermitian symmetry of the UNSYMMETRISED
        condensed matrix, scaled entrywise by the square roots of the corresponding diagonal
        entries, so it is dimensionless and treats the translation and rotation blocks alike;
        an unscaled absolute defect would be blind to a gross asymmetry in the rotational
        block, whose entries are of order one against 1e5 for the translational block. Reject
        when it exceeds 1e-6, and symmetrise only after that check has passed.
        capacity is the (6, 6) Hermitian positive definite matrix and eigenvalues
        its six eigenvalues ascending.
        iterations is a length-six sequence of actual diagonal-preconditioned
        conjugate-gradient iteration counts, ordered by the six boundary fields
        below. Obtain these counts from the six right-hand sides in
        K_ff X = -K_fc E.
        The six columns of E are the six rigid motions of the resonator sampled at the
        surface degrees of freedom, in the order: the three unit Cartesian translations,
        then the three unit rotations about the axes through reference_point. Rotations
        carry radian, so the first three columns are dimensionless and the last three
        carry metre.

    Raises
    ------
    ValueError
        When the three blocks do not have consistent shapes, when surface_dofs does not match the surface block, when n_side is not an integer of four or more, when h is not finite and above zero, when reference_point does not hold exactly three finite values, when rel_tol is not finite and above zero, when the iteration fails to reach rel_tol, when the condensed matrix is not Hermitian to working accuracy, or when the resulting matrix fails to be positive definite.
    """
    return
```

### Step 6

06_subwavelength_frequencies

Goal
----
Return the ordinary frequencies in hertz as well as the angular frequencies, since the frequencies a band diagram is read in are the ordinary ones, and sort ascending so that entry zero is the lowest of the six branches at this quasi-momentum.

Returns
-------
dict, holding the angular and hertz frequency sextuples, the resonator mass and the cross_check_defect.

```python
def subwavelength_frequencies(
    capacity: np.ndarray,
    volume_D: float,
    inertia: np.ndarray,
    rho: float,
    eps: float,
) -> dict:
    """Turn the six by six capacity matrix into the six subwavelength resonant frequencies.

    Parameters
    ----------
    capacity : np.ndarray
        The (6, 6) Hermitian positive definite capacity matrix from the condensation.
    volume_D : float
        Resonator volume in cubic metre, above zero.
    inertia : np.ndarray
        The (6, 6) inertia matrix of the uniform rigid resonator, expressed in the same six-motion
        basis as capacity: the same column order, the same reference point for the rotations. Its
        entries carry the units the basis requires.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    eps : float
        Reciprocal density contrast, above zero and below one.

    Returns
    -------
    dict
        Under the keys angular, hertz, mass and cross_check_defect.
        The generalised problem is capacity v = omega squared times inertia v. Solve that
        generalised problem; the eigenvalues of capacity on their own are not the physical
        spectrum, because the translation columns are dimensionless and the rotation
        columns carry metre.
        mass is the resonator mass rho divided by eps, times volume_D.
        angular and hertz hold the six frequencies ascending, in radian per second and
        in hertz. cross_check_defect is the largest absolute difference in hertz
        between the frequencies formed with the background density and an explicit
        contrast factor and those formed with the resonator density directly.

    Raises
    ------
    ValueError
        When capacity is not a (6, 6) Hermitian positive definite array, when volume_D or rho is not finite and above zero, when inertia is not a (6, 6) finite Hermitian positive definite array, or when eps falls outside the open interval from zero to one.
    """
    return
```

### Step 7

07_dilute_ball_reference

Goal
----
Take r as the radius of the ball of the same volume as the resonator actually used, so that the comparison holds the amount of heavy material fixed and varies only the shape. Treat the interval as a reference comparison and not as a bound. The bracketing statement belongs to a dilute arrangement of balls, and neither changing the shape at fixed volume nor leaving the dilute regime preserves it without a further argument that is not supplied here, so a computed frequency outside the interval would not by itself prove an error and one inside proves nothing.

Returns
-------
dict, holding radius, beta_ball, omega_min, omega_max, hertz_min, hertz_max and width_ratio.

```python
def dilute_ball_reference(
    lam: float,
    mu: float,
    rho: float,
    eps: float,
    volume_D: float,
) -> dict:
    """Evaluate the dilute-limit interval for a ball of the same volume as the resonator.

    Parameters
    ----------
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal, above zero.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    eps : float
        Reciprocal density contrast, above zero and below one.
    volume_D : float
        Resonator volume in cubic metre, above zero.

    Returns
    -------
    dict
        Under the keys radius, beta_ball, omega_min, omega_max, hertz_min, hertz_max and width_ratio.
        width_ratio is the dimensionless endpoint ratio omega_max / omega_min,
        equivalently hertz_max / hertz_min.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when mu, rho or volume_D fails to be above zero, when eps falls outside the open interval from zero to one, or when five mu plus two lam or two mu plus lam fails to be above zero.
    """
    return
```

### Step 8

step_08_assemble_full_bloch_pencil

Goal
----
The mass matrix is the consistent one: the element mass is the density times the integral over the element of the product of the trilinear shape functions, evaluated with the same two-point Gauss rule in each direction as the stiffness, which is exact for that integrand. Assemble it with the same degree of freedom ordering as the stiffness. Every element of the grid carries both a stiffness and a mass, the interior ones scaled as above.

Both matrices are assembled with the same quasi-periodic rule as the exterior stiffness: a corner that wraps across a cell face carries the Bloch phase of the wrap, conjugated on the outgoing side and plain on the incoming, so that both matrices are Hermitian. When every Bloch phase is real the assembled matrices are real and may be kept so. The resonator is placed as the partition placed it, its low corner at element index n_side minus span, integer-divided by two, in each direction.

Returns
-------
dict, holding the sparse Hermitian matrices K_full and M_full over all three n_side cubed degrees of freedom, the count n_inside of elements inside the resonator, and total_mass, the mass of the whole cell.

```python
def assemble_full_bloch_pencil(
    n_side: int,
    spans: tuple,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    h: float,
    rho: float,
    delta: float,
    eps: float,
    alpha: tuple,
) -> dict:
    """Assemble the finite-contrast two-phase stiffness and consistent mass of the whole cell at one quasi-momentum.

    Parameters
    ----------
    n_side : int
        Elements along each edge of the cell, at least four.
    spans : tuple
        Three integers giving the elements spanned by the resonator along each axis.
    element_stiffness : np.ndarray
        The (24, 24) background element stiffness in newton per metre.
    corner_signs : np.ndarray
        The (8, 3) binary corner offsets, in the prescribed order.
    h : float
        Element edge in metre, above zero.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    delta : float
        Reciprocal stiffness contrast, above zero.
    eps : float
        Reciprocal density contrast, above zero.
    alpha : tuple
        The quasi-momentum, three finite values, dimensionless.

    Returns
    -------
    dict
        Under the keys K_full, M_full, n_inside and total_mass.
        K_full and M_full are scipy sparse matrices of shape (3 n_side cubed, 3 n_side cubed),
        Hermitian, with degree of freedom index equal to three times the flat node index plus
        the component, the flat node index being (i times n_side plus j) times n_side plus k
        after periodic wrapping. M_full is the consistent mass. n_inside is the number of
        elements inside the resonator. total_mass is rho times the exterior volume plus rho
        over eps times the resonator volume, in kilogram.

    Raises
    ------
    ValueError
        When n_side is not an integer of four or more, when spans does not hold three integers each at least one and at most n_side minus two, when element_stiffness is not (24, 24) or corner_signs is not (8, 3), when h, rho, delta or eps is not finite and above zero, or when alpha does not hold exactly three finite values.
    """
    return
```

### Step 9

step_09_full_pencil_spectrum

Goal
----
The pencil is large and sparse, and the frequencies wanted sit at the bottom of a spectrum that reaches into the megahertz, so the eigenvalues are to be found by a method suited to that: a shift-and-invert iteration about a low shift, or any other iteration that returns the smallest eigenvalues of a Hermitian positive definite pencil, factorising the sparse matrix once and iterating on the factor. Whatever iteration is used, each returned pair must be checked against the pencil directly, by forming the residual of the eigenvalue equation and requiring it to be small relative to the stiffness term; an iteration that has not converged is rejected rather than reported. When the matrices are real, keep the arithmetic real.

Returns
-------
dict, holding the count lowest eigenfrequencies of the pencil ascending, in hertz and in radian per second, the relative residual of each, and the count.

```python
def full_pencil_spectrum(
    K_full: "scipy.sparse.spmatrix",
    M_full: "scipy.sparse.spmatrix",
    count: int,
    rel_tol: float,
) -> dict:
    """Return the lowest eigenfrequencies of the Hermitian pencil K_full v = omega squared M_full v.

    Parameters
    ----------
    K_full : scipy.sparse.spmatrix
        The finite-contrast stiffness of the whole cell, Hermitian positive definite.
    M_full : scipy.sparse.spmatrix
        The consistent mass of the whole cell, Hermitian positive definite, same shape.
    count : int
        How many of the lowest eigenfrequencies to return, at least one and below the order.
    rel_tol : float
        Bound on the relative residual of each returned pair, finite and above zero.

    Returns
    -------
    dict
        Under the keys hertz, angular, residuals and count. hertz and angular are float arrays
        of shape (count,), ascending, the eigenfrequencies in hertz and in radian per second.
        residuals holds, for each returned pair, the norm of K_full v minus omega squared
        M_full v divided by the norm of K_full v, each at most rel_tol. count is the number
        returned.

    Raises
    ------
    ValueError
        When K_full or M_full is not square, when their shapes differ, when either departs from Hermitian symmetry by more than one part in 1e-9 of its largest entry, when count is not an integer between one and the order minus one, when rel_tol is not finite and above zero, when an eigenvalue comes out non-positive, or when any returned pair fails its residual bound.
    """
    return
```

### Step 10

step_10_report_finite_contrast_correction

Goal
----
Run the leading-order chain at every quasi-momentum in the list, in this order. Step 1 resolves the contrast bookkeeping. Step 2 partitions the cell and returns its edge lengths; it is run once, as is step 3, which forms the element stiffness of one background cube. Then, at each quasi-momentum in turn, step 4 assembles the quasi-periodic exterior stiffness, step 5 condenses it onto the resonator surface and tests the resulting map against all six rigid motions, and step 6 solves the generalised problem against the inertia of the rigid resonator, supplied in the same basis, to give the six branch frequencies. Step 7 supplies the dilute-limit reference.

Then, at the quasi-momentum attaining the leading-order upper edge and there only, run the finite-contrast chain: step 8 assembles the full two-phase pencil of the cell, and step 9 returns its eight lowest eigenfrequencies, each converged to a relative residual below one part in 1e8. The six lowest of those are the finite-contrast counterparts of the six leading-order branches; which of the eight is the first ordinary branch of the cell is for the solver to identify, and it is reported as the ordinary branch. The graded correction is the fourth lowest finite-contrast frequency minus the leading-order upper edge, in hertz.

Report the leading-order quantities alongside: both edges and the width of the leading-order gap, the branch maxima summed over the six branches, the position of the leading-order lower edge within the dilute-limit interval for a ball of the same volume, and the shear wavelength of the background at the leading-order upper edge divided by the lattice constant.

Returns
-------
dict, holding correction_hertz, the graded finite-contrast correction; finite_contrast_edge_hertz and full_pencil_hertz, the corrected edge and the eight lowest finite-contrast frequencies at the attaining quasi-momentum; ordinary_branch_hertz, the first of them that is an ordinary branch of the cell; bandgap_edge_hertz and bandgap_edge_angular for the leading-order upper edge; gap_upper_hertz, gap_lower_hertz, gap_width_hertz and bandwidth_sum_hertz; argmax_index and argmax_branch for the entry attaining the upper edge, lower_argmax_index and lower_argmax_branch for the lower edge; band_tops, frequencies, ball_hertz_min, ball_hertz_max, position_in_interval, tau and wavelength_ratio.

```python
def report_finite_contrast_correction(
    lattice_constant: float,
    n_side: int,
    spans: tuple,
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
    alphas: tuple,
    rel_tol: float,
) -> dict:
    """Scan the prescribed quasi-momenta for the leading-order gap, then correct its upper edge at finite contrast with the full two-phase pencil, and report the correction.

    Parameters
    ----------
    lattice_constant : float
        Edge of the cubic unit cell in metre.
    n_side : int
        Elements along each edge of the cell.
    spans : tuple
        Elements spanned by the resonator along each axis.
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal.
    rho : float
        Background density in kilogram per cubic metre.
    delta : float
        Reciprocal stiffness contrast.
    eps : float
        Reciprocal density contrast.
    alphas : tuple
        Quasi-momenta to scan.
    rel_tol : float
        Relative residual at which each leading-order solve stops. The two-phase
        pencil is solved to the residual bound the prompt states, 1e-8, which
        this argument does not alter.

    Returns
    -------
    dict
        Under the keys correction_hertz, finite_contrast_edge_hertz, full_pencil_hertz,
        ordinary_branch_hertz, bandgap_edge_hertz, bandgap_edge_angular, gap_upper_hertz,
        gap_lower_hertz, gap_width_hertz, bandwidth_sum_hertz, argmax_index, argmax_branch,
        lower_argmax_index, lower_argmax_branch, band_tops, frequencies, ball_hertz_min,
        ball_hertz_max, position_in_interval, tau and wavelength_ratio.
        correction_hertz is the graded quantity: the fourth lowest finite-contrast frequency at
        the quasi-momentum attaining the leading-order upper edge, minus that leading-order
        edge, in hertz. finite_contrast_edge_hertz is that fourth lowest frequency and
        full_pencil_hertz the float array of the eight lowest, ascending; ordinary_branch_hertz
        is the first of them that is an ordinary branch of the cell. bandgap_edge_hertz is the leading-order upper edge and equals
        gap_upper_hertz; argmax_index and argmax_branch locate it, the branch index lying in
        three to five. gap_lower_hertz is the leading-order lower edge, located by
        lower_argmax_index and lower_argmax_branch with that branch index in zero to two, and
        gap_width_hertz is their difference. band_tops holds the largest value each of the six
        leading-order branches attains and frequencies is the (n_alpha, 6) table, sorted
        ascending at each quasi-momentum. position_in_interval places the leading-order LOWER
        edge within the dilute interval, and wavelength_ratio is the background shear
        wavelength at the leading-order upper edge divided by the lattice constant.

    Raises
    ------
    ValueError
        When alphas is empty, when any argument fails the checks the earlier stages impose, or when the scan produces no finite frequency.
    """
    return
```
