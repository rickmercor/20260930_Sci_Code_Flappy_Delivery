# Mathematics-Computational_Mechanics-27

## Background

Computational homogenisation predicts the bulk elastic response of a heterogeneous solid from its microstructure by solving a cell problem under a prescribed macroscopic strain. Solvers built on the fast Fourier transform are efficient because they never assemble a global matrix: the material heterogeneity enters element by element and a constant-coefficient reference operator, which is diagonal in frequency space, is used as a preconditioner. The price is a regular grid, which describes a curved phase boundary only as a staircase.

Extended finite elements describe the boundary independently of the grid. A level set locates the interface, and additional shape functions supported only on cut elements reproduce the kink that the displacement has where the stiffness jumps. The enriched functions must be chosen so that they do not become linearly dependent on the standard ones, and they must be rescaled so that the two blocks of the resulting system are of comparable magnitude, otherwise the conditioning degrades rapidly as an element is cut ever closer to one of its vertices.

Assessing such a scheme requires a configuration whose exact solution is known. A coated sphere whose two radii are tuned so that the pair is neutral has that property under hydrostatic loading: the surrounding matrix is left in the uniform macroscopic state and the interior field is radial and available in closed form, so both the homogenised modulus and the pointwise fields can be compared with the truth at any resolution.

## Problem

Voxel-based Fourier solvers for periodic elastic homogenisation are fast because the grid stays regular, but a curved material interface that cuts the grid is replaced by a staircase, so the local fields lose accuracy where the strain gradient is largest and the homogenised moduli converge only at first order. An interface-enriched discretisation removes the staircase without giving up the regular grid: on cut elements only, it adds basis functions that carry the kink the displacement has across an interface, and it keeps a constant-coefficient preconditioner that can still be applied by fast Fourier transform. Whether such a scheme genuinely delivers interface-conforming accuracy cannot be decided on a microstructure whose exact solution is unknown, so the claim has to be measured against a configuration that has one. Your task is to build that measurement for a coated sphere embedded in a matrix under hydrostatic macroscopic strain, and to report one dimensionless number, namely the relative $L^2$ error of the simulated local strain field against the exact local strain field.

The microstructure has three isotropic phases sharing one Poisson ratio: an inclusion of radius $r_i$, a concentric coating out to radius $r_c$, and the surrounding matrix. The outer radius is prescribed below; the inner radius is not, and must be obtained from the requirement that the coated sphere be neutral, that is that the effective bulk modulus of the cell equal the bulk modulus of the matrix. Neutrality is also what makes the exact solution available: the macroscopic hydrostatic strain then passes through the matrix undisturbed, so the exact displacement is $u_r = r$ for $r \ge r_c$, is radial of the form $u_r = A r + B / r^2$ in the coating, and is $u_r = C r$ in the inclusion, where in a shell of bulk modulus $K$ and shear modulus $\mu$ the volumetric strain is $3 a$ and the radial traction is $\sigma_{rr} = 3 K a - 4 \mu b / r^3$. Continuity of the radial displacement and of the radial traction at the two interfaces gives four conditions for the three constants; three of them determine $A$, $B$ and $C$, and the residual of the fourth is the check that the geometry and the moduli are mutually consistent.

Discretise the cell by equal cubic voxels, each split into the six prescribed linear tetrahedra listed below, with periodic mean-free nodal displacement fluctuations. A single scalar level set encodes both interfaces as the signed distance to the nearer one, taken negative in the coating, and the discrete interface is the zero set of its piecewise linear interpolant. Every node of a cut element carries three extra degrees of freedom whose scalar shape functions are $N_i \rho^m$, built from the modified absolute enrichment $\rho^m = \sum_i N_i |L_i| - |\sum_i N_i L_i|$, and each enriched column is internally scaled so that its reference energy is one. Solve the resulting equilibrium system by linear conjugate gradients from a zero fluctuation, preconditioned by the block matrix whose enriched block is the identity and whose standard block is the constant-coefficient operator $\int_Y \nabla^s N_i : \nabla^s N_j \, dV$, applied through its Fourier multiplier. Then evaluate both the simulated and the exact strain at the quadrature points of the assembled rule and form the relative error.

Use the following deterministic configuration:

- cell $Y = [0, 16]^3$ in micrometres, periodic in all three directions, with mean-free standard nodal displacement fluctuations
- `n_voxels = 16` along each axis, so the mesh parameter is $h = 1$ micrometre
- local cube vertices `[(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]`
- local tetrahedra `[(0,1,2,6), (0,2,3,6), (0,3,7,6), (0,7,4,6), (0,4,5,6), (0,5,1,6)]`, indexing that vertex list; this subdivision is prescribed and no other splitting of the voxel into six tetrahedra may be substituted, because the reported number depends on it in its fourth significant figure
- the coated sphere is centred at $c = (8, 8, 8)$ micrometres and the coating radius is $r_c = 2 \pi$ micrometres
- the inclusion radius $r_i$ is not given: it is the radius at which the effective bulk modulus of the cell equals the bulk modulus of the matrix
- bulk moduli in MPa: matrix $1.000000$, coating $0.808024$, inclusion $8.080240$; Poisson ratio $0.25$ in all three phases
- phase membership by radius: inclusion where $\|x - c\| < r_i$, coating where $r_i \le \|x - c\| < r_c$, matrix where $\|x - c\| \ge r_c$; the coated sphere lies strictly inside the cell, so no periodic image is involved
- level set: with $r = \|x - c\|$ and $r_{mid} = (r_i + r_c) / 2$, take $L(x) = r - r_c$ where $r \ge r_{mid}$ and $L(x) = r_i - r$ where $r < r_{mid}$, so that $L$ is negative in the coating and positive in both the matrix and the inclusion
- $L_h$ is the piecewise linear interpolant of the nodal values of $L$; an element is cut when its four nodal values do not share a sign, and a node is enriched when it belongs to at least one cut element; no nodal value lies within $10^{-6}$ micrometres of zero, and no element meets both interfaces
- macroscopic Mandel strain $\bar{\varepsilon} = (1, 1, 1, 0, 0, 0)$ in orthonormal Mandel order $(11, 22, 33, 23, 13, 12)$, and the total strain is $\varepsilon = \bar{\varepsilon} + \nabla^s u$
- the displacement fluctuation is the field for which $\int_Y \nabla^s v : C : \varepsilon \, dV = 0$ holds for every periodic test field $v$, so the macroscopic strain enters the discrete system as a load of the opposite sign to $\sum_\alpha w_\alpha B_e^{T} C \bar{\varepsilon}$
- quadrature: every element is integrated by the symmetric four-point degree-two rule applied to each of its sub-tetrahedra, an uncut element counting as its own single sub-tetrahedron; a cut element is divided into the minimum number of sub-tetrahedra that resolve $L_h = 0$, which is four when one vertex is separated from the other three and six when two are separated from two, and a resulting prism with matched triangular faces $(t_0, t_1, t_2)$ and $(u_0, u_1, u_2)$ is split as $(t_0, t_1, t_2, u_0)$, $(t_1, t_2, u_0, u_1)$, $(t_2, u_0, u_1, u_2)$; the whole cell then carries 160176 quadrature points
- the four-point rule uses the barycentric permutations of $(a, b, b, b)$ with $a = (5 + 3\sqrt{5}) / 20$ and $b = (5 - \sqrt{5}) / 20$, and every weight is one quarter of the sub-tetrahedron volume
- the stiffness at a quadrature point is that of the coating where $L_h < 0$, and otherwise that of the matrix if the element has a matrix node and of the inclusion if it has an inclusion node
- internal scaling is componentwise: for enriched node $j$ and displacement direction $a$ the factor is $D_{ja} = \int_Y \| \nabla^s (N_j \rho^m e_a) \|^2 \, dV$, giving one factor per node and per direction rather than one per node, and the corresponding enriched column of the strain operator is divided by $\sqrt{D_{ja}}$
- the constant-coefficient reference stiffness is the $6 \times 6$ identity in Mandel space, the Fourier corner phase is $\exp(2 \pi i \, Y_j \cdot k / n)$ over the eight voxel corners, and the multiplier at zero frequency is zero; the standard block of the preconditioner must be applied through this Fourier multiplier
- the macroscopic stress $\langle \sigma \rangle_Y$ is the volume average over the cell, not the volume integral, and the effective bulk modulus is one ninth of the trace of its axial part
- linear conjugate gradients start from a zero fluctuation; the monitored residual is $\mathrm{res}_k = \sqrt{r_k^{T} \widetilde{P}^{-1} r_k}$ and the stopping test is $\mathrm{res}_k \le \mathrm{tol} \, \| \langle \sigma \rangle_Y \|_2$ with `tol = 1e-7` and `max_iter = 200`
- the exact strain at a quadrature point is evaluated from the shell law of the phase already assigned to that point for the stiffness, that is the phase implied by the linearised interface, and not from the phase implied by the true spherical radius
- both norms use the same quadrature points and weights, $\| a \|_{L^2} = ( \sum_\alpha w_\alpha \, a(q_\alpha) \cdot a(q_\alpha) )^{1/2}$, so any common volume normalisation cancels
- IEEE float64 arithmetic and NumPy's default fast Fourier transform normalisation

State in your reasoning the derived inclusion radius in micrometres, the three exact shell coefficients, the residual of the interface condition you did not use to determine them, the number of cut elements and the number of enriched nodes, the quadrature-weighted volume of the inclusion phase in cubic micrometres, and the simulated effective bulk modulus in MPa. Report as the final answer the relative $L^2$ strain error $e = \| \varepsilon_* - \varepsilon_h \|_{L^2} / \| \varepsilon_* \|_{L^2}$ to at least six significant figures; it is graded to a relative tolerance of $10^{-4}$.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_derive_neutral_inclusion_radius

Goal
----
This step fixes the geometry of the benchmark before any discretisation is considered. From the three bulk moduli, the shared Poisson ratio and the outer coating radius it returns the inclusion volume fraction that renders the coated sphere neutral, the inclusion radius that fraction implies, and the equivalent bulk modulus evaluated at that fraction.

The third quantity is not consumed downstream. It is returned because neutrality requires it to reproduce the matrix bulk modulus to machine precision, so it is the cheapest available evidence that the closed form and the neutrality condition have been paired correctly rather than merely being plausible.

```python
def derive_neutral_inclusion_radius(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
) -> dict:
    """Solve the neutrality condition for the inclusion radius of a coated sphere.

    Parameters
    ----------
    bulk_matrix : float
        Bulk modulus of the matrix, strictly positive.
    bulk_coating : float
        Bulk modulus of the coating, strictly positive.
    bulk_inclusion : float
        Bulk modulus of the inclusion, strictly positive.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_coating : float
        Outer radius of the coating, strictly positive.

    Returns
    -------
    dict
        Keys volume_fraction, radius_inclusion and effective_bulk.

    Raises
    ------
    ValueError
        If any bulk modulus is not strictly positive, if poisson_ratio lies outside the open interval (-1, 1/2), if radius_coating is not strictly positive, if the coating and inclusion moduli coincide so that no neutral radius exists, or if the neutrality condition has no admissible root in (0, 1).
    """
    return
```

### Step 2

02_solve_coated_sphere_exact_field

Goal
----
This step solves the three-by-three interface system for the shell coefficients of the exact radial field, and evaluates the fourth, unused, interface condition as a residual. The coefficients are the reference solution against which the discrete strain field is measured at the end of the chain, so they must be obtained from the interface conditions themselves rather than from any approximation of them.

The residual is reported alongside them. It is not an error estimate of this step: it is a consistency test on the pair of steps that precede the mesh, because it can vanish only when the inclusion radius handed in really does make the coated sphere neutral.

```python
def solve_coated_sphere_exact_field(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Solve the three-shell interface system for the exact radial field.

    Parameters
    ----------
    bulk_matrix : float
        Bulk modulus of the matrix.
    bulk_coating : float
        Bulk modulus of the coating.
    bulk_inclusion : float
        Bulk modulus of the inclusion.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_inclusion : float
        Inner interface radius, strictly positive and smaller than radius_coating.
    radius_coating : float
        Outer interface radius.

    Returns
    -------
    dict
        Keys coefficient_a, coefficient_b, coefficient_c and traction_residual.

    Raises
    ------
    ValueError
        If the radii do not satisfy 0 < radius_inclusion < radius_coating, or if poisson_ratio lies outside the open interval (-1, 1/2).
    """
    return
```

### Step 3

03_build_three_phase_periodic_mesh

Goal
----
This step builds the periodic tetrahedral mesh, evaluates the nodal level set on it, and derives from those two the bookkeeping that every later step reads: which elements are cut, which phase lies on the positive side of each cut, which nodes carry enrichment and how many degrees of freedom the enriched system therefore has.

The single-interface assumption is enforced rather than assumed. An element whose nodes include both a matrix node and an inclusion node would need two interfaces inside one tetrahedron, which the subdivision of the next step cannot represent, so such a configuration is rejected as invalid. The smallest absolute nodal level set value is returned as well, because an interface that passes very close to a node is what degrades the conditioning that the internal scaling is later designed to repair.

```python
def build_three_phase_periodic_mesh(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    radius_inclusion: float,
    radius_coating: float,
) -> dict:
    """Build the periodic tetrahedral mesh and the level set data of a coated sphere.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge, at least two.
    cell_size : float
        Edge length of the cubic periodic cell, strictly positive.
    centre : tuple
        Three coordinates of the centre of the coated sphere.
    radius_inclusion : float
        Radius of the inner interface.
    radius_coating : float
        Radius of the outer interface.

    Returns
    -------
    dict
        Keys element_vertices, element_nodes, element_levels, is_cut, element_outer_phase,
        enriched_index, n_nodes, n_elements, n_enriched, n_dof and smallest_absolute_level.

    Raises
    ------
    ValueError
        If n_voxels is below two, if cell_size is not strictly positive, if the radii do not satisfy 0 < radius_inclusion < radius_coating, or if any element touches both interfaces, which makes the configuration invalid for single-interface subdivision.
    """
    return
```

### Step 4

04_build_interface_subcell_quadrature

Goal
----
This step returns the quadrature of one tetrahedron: the barycentric coordinates of its points, their weights, and the side of the discrete interface on which each point lies. The points are reported in barycentric rather than Cartesian coordinates because the shape functions and the enrichment are both evaluated from them, and because the caller already holds the vertices.

The side label is the reason the rule is built at all. It is what allows the later assembly to give a point the coating stiffness or the positive-side stiffness without ever asking for the true spherical radius, so the material data seen by the quadrature is consistent with the linearised interface that the subdivision has just resolved.

A vanishing nodal level set value is rejected. It leaves a sub-tetrahedron of zero volume and a crossing point that coincides with a vertex, so the decomposition is degenerate rather than merely awkward.

```python
def build_interface_subcell_quadrature(vertices, levels) -> dict:
    """Build the interface-resolving quadrature of one linear tetrahedron.

    Parameters
    ----------
    vertices : array_like
        Array of shape (4, 3) holding the tetrahedron vertices.
    levels : array_like
        Array of shape (4,) holding the nodal level set values.

    Returns
    -------
    dict
        Keys barycentric of shape (q, 4), weights of shape (q,) and side of shape (q,).

    Raises
    ------
    ValueError
        If vertices does not have shape (4, 3) or levels does not have shape (4,), or if a nodal level set value vanishes, which leaves the interface unresolved.
    """
    return
```

### Step 5

05_evaluate_modified_abs_enrichment

Goal
----
This step evaluates the modified enrichment and its gradient at a supplied set of points of one element, from the nodal level set values and the constant gradients of the four linear shape functions.

Both quantities are returned because the enriched strain operator needs both: the value multiplies the shape function gradient and the gradient multiplies the shape function, and the two terms are of comparable size on a barely cut element. The gradient is evaluated phasewise, from the sign of the interpolated level set at each point, rather than by differentiating a smoothed surrogate, so the jump across the interface is represented exactly and not spread over a layer of points.

```python
def evaluate_modified_abs_enrichment(barycentric, levels, gradients) -> dict:
    """Evaluate the modified absolute enrichment and its gradient at given points.

    Parameters
    ----------
    barycentric : array_like
        Array of shape (q, 4) of barycentric coordinates.
    levels : array_like
        Array of shape (4,) of nodal level set values.
    gradients : array_like
        Array of shape (4, 3) of linear shape function gradients.

    Returns
    -------
    dict
        Keys rho of shape (q,) and grad_rho of shape (q, 3).

    Raises
    ------
    ValueError
        If barycentric does not have shape (q, 4), or if levels does not have shape (4,) or gradients does not have shape (4, 3).
    """
    return
```

### Step 6

06_assemble_scaled_three_phase_system

Goal
----
This step turns the mesh and the three phase stiffnesses into the element operators the solver will use, and into the single global array of quadrature points, weights and phases that the error functional will later read.

It performs two passes over the elements, and the order matters. The first pass evaluates the unscaled enriched columns and accumulates the reference energy of each enriched node and direction, because a scaling factor is an integral over every element that shares the node and cannot be known from one element alone. The second pass rebuilds the columns with the factors applied and forms the element stiffness and stress map.

The step calls the interface quadrature step and the enrichment step for every element rather than repeating their logic. Inactive columns, those of enriched slots on uncut elements, are zeroed and directed at a discard index, so the later matrix-free product may sum over all twenty-four local entries without testing whether each one exists.

```python
def assemble_scaled_three_phase_system(mesh: dict, stiffness) -> dict:
    """Assemble the internally scaled element operators of the enriched system.

    Parameters
    ----------
    mesh : dict
        Mesh description as returned by the mesh construction step.
    stiffness : array_like
        Array of shape (3, 6, 6) of Mandel stiffness matrices ordered matrix, coating, inclusion.

    Returns
    -------
    dict
        Keys element_stiffness, element_stress_map, element_dofs, scaling, integrated_stiffness, quadrature_points, quadrature_weights, quadrature_phase, strain_operator, quadrature_element, n_dof, n_nodes and n_quadrature.

    Raises
    ------
    ValueError
        If stiffness does not have shape (3, 6, 6).
    """
    return
```

### Step 7

07_build_fourier_green_operator

Goal
----
This step builds the multiplier once, for a given voxel count and cell edge, so that the solver can apply the inverse of the constant-coefficient operator by a forward transform, a pointwise product and an inverse transform.

Only one voxel is assembled. Every voxel of the periodic mesh is a translate of that reference voxel, which is what makes the operator a convolution, so the whole preconditioner is determined by a matrix of twenty-four rows and twenty-four columns together with the corner offsets. The multiplier carries no material data: it is a preconditioner, so its quality but not the solution it converges to depends on how well it resembles the real operator.

```python
def build_fourier_green_operator(n_voxels: int, cell_size: float):
    """Build the block Fourier multiplier of the inverse constant-coefficient operator.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge, at least two.
    cell_size : float
        Edge length of the cubic periodic cell, strictly positive.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (n_voxels, n_voxels, n_voxels, 3, 3).

    Raises
    ------
    ValueError
        If n_voxels is below two, or if cell_size is not strictly positive.
    """
    return
```

### Step 8

08_solve_scaled_xfft_system

Goal
----
This step solves the equilibrium system for one prescribed macroscopic strain and returns the converged fluctuation together with the diagnostics that certify the solve: the iteration count, the history of the preconditioned residual, the threshold in force at the end, the volume-averaged Mandel stress and a convergence flag.

The stopping threshold is not a constant of the run. It is recomputed from the current volume-averaged stress at every iteration, so the test measures the residual against the quantity the calculation exists to produce rather than against the initial residual, and a configuration whose macroscopic stress is small is not thereby granted a looser solve.

The averaged stress is assembled from the integrated stiffness acting on the macroscopic strain plus the element stress maps acting on the fluctuation, and divided by the cell volume. It is therefore available at every iteration at the cost of one pass over the elements, which is what makes the running threshold affordable.

```python
def solve_scaled_xfft_system(
    operators: dict,
    green_operator,
    macroscopic_strain,
    cell_volume: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Solve the preconditioned enriched equilibrium system by conjugate gradients.

    Parameters
    ----------
    operators : dict
        Assembled element operators.
    green_operator : array_like
        Complex Fourier multiplier of shape (n, n, n, 3, 3).
    macroscopic_strain : array_like
        Prescribed macroscopic strain of shape (6,) in Mandel order.
    cell_volume : float
        Volume of the periodic cell.
    tolerance : float
        Relative tolerance of the stopping test.
    max_iterations : int
        Maximum number of conjugate-gradient iterations.

    Returns
    -------
    dict
        Keys displacement, iterations, residuals, threshold, effective_stress and converged.

    Raises
    ------
    ValueError
        If macroscopic_strain does not have shape (6,), if cell_volume is not strictly positive, or if tolerance is not positive or max_iterations is below one.
    """
    return
```

### Step 9

09_compute_relative_l2_strain_error

Goal
----
This step is the orchestrator. It runs the whole chain in sequence: step 1 supplies the inclusion radius from the neutrality condition, step 2 the exact shell coefficients, step 3 the periodic three-phase mesh and its level set, steps 4 and 5 the interface quadrature and the modified enrichment that step 6 uses to assemble the internally scaled element operators, step 7 the Fourier multiplier of the preconditioner and step 8 the conjugate-gradient solution of the equilibrium system. It then compares the discrete and exact strain fields at every quadrature point of the assembled rule and forms the relative error.

It returns the error together with the intermediate quantities a reader needs to judge it: the simulated effective bulk modulus, which neutrality requires to return the matrix value, the derived inclusion radius, the shell coefficients and the residual of the interface condition that was not used to determine them, the mesh and quadrature counts, the quadrature-weighted volume of each phase and the iteration count of the solve.

```python
def compute_relative_l2_strain_error(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
    macroscopic_strain,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Run the enriched solver and return the relative local strain error.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge.
    cell_size : float
        Edge length of the cubic periodic cell.
    centre : tuple
        Centre of the coated sphere.
    bulk_matrix : float
        Bulk modulus of the matrix.
    bulk_coating : float
        Bulk modulus of the coating.
    bulk_inclusion : float
        Bulk modulus of the inclusion.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_coating : float
        Outer radius of the coating.
    macroscopic_strain : array_like
        Prescribed macroscopic strain of shape (6,).
    tolerance : float
        Relative tolerance of the stopping test.
    max_iterations : int
        Maximum number of conjugate-gradient iterations.

    Returns
    -------
    dict
        Keys relative_error, effective_bulk_modulus, radius_inclusion, coefficients, traction_residual, n_cut, n_enriched, n_quadrature, phase_volumes, iterations, error_norm_squared and exact_norm_squared.

    Raises
    ------
    ValueError
        If macroscopic_strain does not have shape (6,), or if n_voxels is below two or cell_size is not strictly positive.
    """
    return
```
