# Mathematics-Computational_Mechanics-33

## Background

Computational homogenization predicts the bulk elastic response of a heterogeneous material from its microstructure. Periodic FFT solvers are efficient on regular grids, but voxelized descriptions replace curved phase boundaries with staircases, degrading local fields and effective properties where abrupt material changes create sharp strain gradients.

Interface-enriched discretizations represent material boundaries independently of the background grid and add approximation functions only near cut cells. The work combines this geometric treatment with Fourier acceleration while controlling the conditioning problems introduced by enrichment.

The resulting method produces microscale displacement and stress fields together with homogenized elastic quantities for multiphase solids. Its performance is assessed through local and effective accuracy, convergence under mesh refinement, and solver iteration counts across different interface geometries and material contrasts.

## Problem

Classical voxel FFT discretizations of elastic composites lose local and effective accuracy when a smooth material interface cuts the grid. Compute one toy periodic homogenization problem with an interface-enriched, FFT-accelerated P1 finite-element scheme whose modified absolute enrichment and internally scaled strongly stable preconditioner retain an interface-conforming displacement kink. The primary inputs are a spherical signed-distance field and two isotropic stiffnesses, and the output is the effective axial stiffness under unit macroscopic axial strain.

Use a unit cell with three voxels per axis, periodic nodal displacement fluctuations, and six tetrahedra per voxel. Represent symmetric tensors in orthonormal Mandel order `(11, 22, 33, 23, 13, 12)`, resolve the planar level-set interface inside every cut tetrahedron by exact degree-two subcell quadrature, and use the source method's modified absolute enrichment, componentwise internal scaling, and block Fourier preconditioner. Solve the scaled equilibrium system by matrix-free linear conjugate gradients from zero displacement, applying the standard block through the cached Fourier Green operator and the scaled enrichment block through its source-prescribed treatment.

Use the following deterministic configuration:

- cell = `[0, 1]^3`
- `n_voxels = 3`
- local cube vertices = `[(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]`
- local tetrahedra = `[(0,1,2,6), (0,2,3,6), (0,3,7,6), (0,7,4,6), (0,4,5,6), (0,5,1,6)]`
- sphere center `c = (0.43, 0.37, 0.52)` and radius `r = 0.31`
- periodic level set `L(x) = ||min(|x - c|, 1 - |x - c|)||_2 - r`
- inclusion is `L < 0`; matrix is `L > 0`; no nodal value is within `1e-12` of zero
- matrix Lame pair `(lambda_m, mu_m) = (3.0, 2.0)`
- inclusion Lame pair `(lambda_i, mu_i) = (30.0, 20.0)`
- macroscopic Mandel strain `epsilon_bar = (1, 0, 0, 0, 0, 0)`
- sub-tetrahedron quadrature barycentric permutations of `(a, b, b, b)`, where `a = (5 + 3*sqrt(5))/20`, `b = (5 - sqrt(5))/20`, and every weight is one quarter of the sub-tetrahedron volume
- constant-coefficient reference stiffness is the `6 x 6` identity in Mandel space
- Fourier corner phase is `exp(2*pi*i*Y_j dot k/n_voxels)` and the zero-frequency inverse is zero
- `tol = 1e-10`, `max_iter = 500`, IEEE float64 arithmetic, and NumPy's default FFT normalization

Accumulate the volume-averaged Mandel stress from the converged enriched field; because the cell volume and imposed `epsilon_bar_11` are both one, your final answer must be a single number: `C_eff,1111 = sigma_eff,11`. In the reasoning, report the number of periodic nodes, tetrahedra, cut tetrahedra and enriched nodes, the sum of the quadrature weights, the reconstructed inclusion volume, the minimum and maximum internal scaling energy, the scaled load norm, the pre-equilibrium constitutive average stress, the Green multiplier at `k = (1, 1, 1)`, the iteration count with the final preconditioned residual and its threshold, and the six-component effective Mandel stress.

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

01_build_periodic_sphere_mesh

Goal
----
Periodic voxel geometry and interface data for the X-FEM discretization.




A cuboid periodic cell is partitioned into voxels and each voxel into six P1 tetrahedra around the body diagonal. The level set at every physical voxel corner is the periodic signed distance to a sphere. A tetrahedron is enriched exactly when its four nodal level-set values contain both signs, and all global periodic nodes incident to those tetrahedra carry enriched displacement degrees of freedom.




Inputs

------

n_voxels : int

    Number of voxels on every cell edge.

radius : float

    Sphere radius in a unit periodic cell.

center : array-like of shape (3,)

    Sphere center in cell coordinates.




Returns

-------

mesh : dict

    Keys nodes, vertices, node_ids, levels, cut, enriched_nodes, n_voxels, radius, and center, with shapes and dtypes specified below.

```python
import numpy as np


def build_periodic_sphere_mesh(
    n_voxels: int,
    radius: float,
    center: np.ndarray,
) -> dict:
    """Build the periodic tetrahedral mesh and signed-distance interface data.

    Parameters
    ----------
    n_voxels : int
        Number of equal voxels along each axis, at least two.
    radius : float
        Sphere radius, strictly between zero and one half.
    center : np.ndarray
        Sphere center with three coordinates strictly inside the unit cell.

    Returns
    -------
    mesh : dict
        Keys nodes (n_voxels**3, 3) float64, vertices (6*n_voxels**3, 4, 3)
        float64, node_ids (6*n_voxels**3, 4) int, levels (6*n_voxels**3, 4)
        float64, cut (6*n_voxels**3,) bool, enriched_nodes (k,) int, and the
        native n_voxels, radius and center.
    Raises
    ------
    ValueError
        If n_voxels is not an integer of at least two, if radius is not finite
        or not strictly between 0 and 0.5, if center is not a finite array of
        shape (3,), or if any center coordinate lies on or outside the open
        unit cell (0, 1).
    """
    return
```

### Step 2

02_construct_subcell_quadrature

Goal
----
Interface-resolving subcell quadrature for a P1 tetrahedron.




The nodal level-set interpolant is planar inside a P1 tetrahedron. Clipping the tetrahedron by its zero plane yields one convex polyhedron per material phase. Each phase polyhedron is tetrahedralized from an interior centroid, and the resulting sub-tetrahedra use the symmetric four-point degree-two rule needed to integrate the quadratic enriched stiffness integrand exactly.




Inputs

------

vertices : np.ndarray of shape (4, 3)

levels : np.ndarray of shape (4,)




Returns

-------

quadrature : dict

    Original-tetrahedron barycentric points, physical weights, and phase signs.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def construct_subcell_quadrature(
    vertices: np.ndarray,
    levels: np.ndarray,
) -> dict:
    """Construct exact degree-two quadrature on the two phase subdomains.

    Parameters
    ----------
    vertices : np.ndarray
        Physical tetrahedron vertices with shape (4, 3).
    levels : np.ndarray
        Nodal values of the linear level-set interpolant with shape (4,).

    Returns
    -------
    quadrature : dict
        Arrays named barycentric, weights, and phases.
    Raises
    ------
    ValueError
        If vertices is not shape (4, 3) or levels is not shape (4,), if either
        contains a non-finite value, if the tetrahedron volume is at most
        1e-14 (degenerate), or if any nodal level satisfies |level| <= 1e-12
        (an interface node, excluded by the benchmark).
    """
    return
```

### Step 3

03_evaluate_modified_abs_enrichment

Goal
----
Modified absolute enrichment and its physical gradient.




For linear shape functions N_i and nodal signed distances L_i, the modified absolute enrichment subtracts the absolute value of the interpolated level set from the interpolation of the nodal absolute values. Its product with each N_i is the local enriched scalar shape function. The gradient is evaluated separately on each side of the planar interface, where the sign of the interpolated level set is constant.




Inputs

------

barycentric : np.ndarray of shape (q, 4)

levels : np.ndarray of shape (4,)

shape_gradients : np.ndarray of shape (4, 3)




Returns

-------

enrichment : dict

    Enrichment values, gradients, and gradients of all four enriched shapes.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_modified_abs_enrichment(
    barycentric: np.ndarray,
    levels: np.ndarray,
    shape_gradients: np.ndarray,
) -> dict:
    """Evaluate the modified absolute enrichment at quadrature points.

    Parameters
    ----------
    barycentric : np.ndarray
        Original-tetrahedron barycentric points with shape (q, 4).
    levels : np.ndarray
        Nodal signed-distance values with shape (4,).
    shape_gradients : np.ndarray
        Physical gradients of the four P1 shape functions with shape (4, 3).

    Returns
    -------
    enrichment : dict
        Arrays rho, grad_rho, and grad_enriched_shapes.
    Raises
    ------
    ValueError
        If barycentric is not two-dimensional with four columns, if levels is
        not shape (4,) or shape_gradients is not shape (4, 3), if any
        barycentric row does not sum to one (atol 1e-12) or has an entry below
        -1e-12, if the four shape gradients do not sum to zero (atol 1e-12),
        or if any quadrature point lies on the interface (|interpolated
        level| <= 1e-14).
    """
    return
```

### Step 4

04_scale_and_assemble_elements

Goal
----
Internal enrichment scaling and cached X-FEM element operators.




The displacement gradient is represented in orthonormal Voigt-Mandel coordinates. Each enriched displacement component is scaled by the inverse square root of its cell-integrated symmetric-gradient energy, so its unit-material energy norm is one. A second quadrature pass assembles scaled 24 by 24 local matrices, macroscopic-strain loads, and stress maps. These element arrays support later matrix-free gathering and scattering.




Inputs

------

mesh : dict

    Required fields are float64 arrays nodes (v, 3), vertices (e, 4, 3), and levels (e, 4); integer arrays node_ids (e, 4) and enriched_nodes (k,); and Boolean array cut (e,), where v, e, and k count nodes, elements, and enriched nodes.

quadrature : dict

    Required fields are integer arrays element_ptr (e + 1,) and phases (q,), float64 array barycentric (q, 4), and float64 array weights (q,), where q is the total number of quadrature points.

matrix_lame : array-like of shape (2,)

inclusion_lame : array-like of shape (2,)

macrostrain : np.ndarray of shape (6,)




Returns

-------

elements : dict

    Keys scaling_energy, element_matrices, element_rhs, element_stress, element_dofs, rhs, stress_macro, n_standard_dofs, and n_enriched_dofs, with shapes and dtypes specified below.

```python
def scale_and_assemble_elements(
    mesh: dict,
    quadrature: dict,
    matrix_lame: np.ndarray,
    inclusion_lame: np.ndarray,
    macrostrain: np.ndarray,
) -> dict:
    """Apply internal scaling and assemble the matrix-free element cache.

    Parameters
    ----------
    mesh : dict
        Required fields are float64 arrays nodes (v, 3), vertices (e, 4, 3), and levels (e, 4); integer arrays node_ids (e, 4) and enriched_nodes (k,); and Boolean array cut (e,), where v, e, and k count nodes, elements, and enriched nodes.
    quadrature : dict
        Required fields are integer arrays element_ptr (e + 1,) and phases (q,), float64 array barycentric (q, 4), and float64 array weights (q,), where q is the total number of quadrature points.
    matrix_lame : np.ndarray
        Matrix Lamé pair (lambda, mu).
    inclusion_lame : np.ndarray
        Inclusion Lamé pair (lambda, mu).
    macrostrain : np.ndarray
        Prescribed Mandel strain vector.

    Returns
    -------
    elements : dict
        Keys scaling_energy, element_matrices, element_rhs, element_stress, element_dofs, rhs, stress_macro, n_standard_dofs, and n_enriched_dofs, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If mesh or quadrature is not a dict carrying every required field, if
        mesh vertices is not shape (e, 4, 3) or node_ids, levels or cut do not
        align with it, if element_ptr is not shape (e + 1,), does not start at
        0 or does not end at the number of quadrature points, if either Lame
        pair is not shape (2,) with strictly positive entries, if macrostrain
        is not shape (6,), or if any enriched component has scaling energy at
        most 1e-14.
    """
    return
```

### Step 5

05_build_fourier_green_operator

Goal
----
Fourier-space inverse of the homogeneous periodic P1 operator.




The X-FFT block preconditioner uses the constant-coefficient standard finite-element operator and an identity block for scaled enrichment degrees of freedom. Six P1 tetrahedra form the reference voxel. For every nonzero discrete frequency, voxel corner phase factors reduce its 24 by 24 stiffness to a 3 by 3 Hermitian symbol whose pseudoinverse is Green's operator. The zero-frequency symbol is set to zero to enforce the mean-free displacement fluctuation.




Inputs

------

n_voxels : int

    Number of periodic voxels on each cell edge.




Returns

-------

green : np.ndarray of shape (n, n, n, 3, 3)

    Complex Fourier multipliers for the standard displacement block.

```python
import numpy as np


def build_fourier_green_operator(n_voxels: int) -> np.ndarray:
    """Build the cached Fourier Green operator for the standard FE block.

    Parameters
    ----------
    n_voxels : int
        Number of equal periodic voxels along each axis, at least two.

    Returns
    -------
    green : np.ndarray
        Complex array of 3 by 3 inverse symbols at all discrete frequencies.
    Raises
    ------
    ValueError
        If n_voxels is not an integer of at least two.
    """
    return
```

### Step 6

06_solve_scaled_xfft_system

Goal
----
Matrix-free preconditioned linear conjugate gradients for the scaled X-FFT system.




Element stiffness actions are gathered and scattered without assembling a global matrix. The standard residual block is transformed by a three-dimensional FFT, multiplied by the cached Green operator, and inverse transformed. The internally scaled enriched block is preconditioned by the identity. The paper's energy residual is the square root of the preconditioned residual product, and convergence compares it with the norm of the current volume-averaged Mandel stress.




Inputs

------

elements : dict

    Required fields are float64 arrays element_matrices (e, 24, 24), element_stress (e, 6, 24), rhs (d,), and stress_macro (6,); integer array element_dofs (e, 24); and native int n_standard_dofs, where e and d count elements and total degrees of freedom.

green : np.ndarray of shape (n, n, n, 3, 3)

tol : float

max_iter : int




Returns

-------

solution : dict

    Keys displacement, residual_trace, iterations, converged, and effective_stress, with shapes and dtypes specified below.

```python
import numpy as np


def solve_scaled_xfft_system(
    elements: dict,
    green: np.ndarray,
    tol: float,
    max_iter: int,
) -> dict:
    """Solve the scaled X-FFT equilibrium system by preconditioned linear CG.

    Parameters
    ----------
    elements : dict
        Required fields are float64 arrays element_matrices (e, 24, 24), element_stress (e, 6, 24), rhs (d,), and stress_macro (6,); integer array element_dofs (e, 24); and native int n_standard_dofs, where e and d count elements and total degrees of freedom.
    green : np.ndarray
        Fourier inverse symbols for the standard displacement block.
    tol : float
        Positive relative residual tolerance.
    max_iter : int
        Positive maximum iteration count.

    Returns
    -------
    solution : dict
        Keys displacement, residual_trace, iterations, converged, and effective_stress, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If elements is not a dict carrying element_matrices, element_dofs,
        element_stress, rhs, stress_macro and n_standard_dofs, if green is not
        five-dimensional with trailing shape (3, 3), if tol is not a finite
        positive number, if max_iter is not a positive integer, if
        n_standard_dofs differs from 3 * n_voxels**3 implied by green, or if
        the conjugate-gradient iteration meets non-positive curvature.
    """
    return
```

### Step 7

07_compute_effective_axial_stress

Goal
----
Effective stress extraction from the converged enriched displacement field.




Each cached element stress matrix maps its standard and scaled enriched nodal displacements to the integral of the local Mandel stress. Adding those fluctuation contributions to the macroscopic-strain stress integral gives the cell average because the benchmark cell has unit volume. Under unit axial Mandel strain, the first component is the effective axial stiffness C_eff,1111.




Inputs

------

element_stress : np.ndarray of shape (e, 6, 24)

element_dofs : np.ndarray of shape (e, 24)

stress_macro : np.ndarray of shape (6,)

displacement : np.ndarray of shape (d,)




Returns

-------

effective : dict

    Keys effective_stress and effective_axial_stiffness, with shapes and dtypes specified below.

```python
import numpy as np


def compute_effective_axial_stress(
    element_stress: np.ndarray,
    element_dofs: np.ndarray,
    stress_macro: np.ndarray,
    displacement: np.ndarray,
) -> dict:
    """Accumulate the effective Mandel stress and axial stiffness.

    Parameters
    ----------
    element_stress : np.ndarray
        Cached element stress maps with shape (e, 6, 24).
    element_dofs : np.ndarray
        Global degree-of-freedom maps, with -1 for inactive entries.
    stress_macro : np.ndarray
        Integrated stress due to the prescribed macroscopic strain.
    displacement : np.ndarray
        Converged standard and scaled enriched displacement vector.

    Returns
    -------
    effective : dict
        Keys effective_stress and effective_axial_stiffness, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If element_stress is not shape (e, 6, 24), if element_dofs is not
        shape (e, 24), if stress_macro is not shape (6,) or displacement is
        not one-dimensional, or if any non-negative dof index is out of range
        for displacement.
    """
    return
```

### Step 8

08_run_xfft_pipeline

Goal
----
Complete deterministic X-FFT homogenization pipeline.




The orchestrator builds the periodic spherical interface, resolves every cut tetrahedron by subcell quadrature, evaluates the modified absolute enrichment, applies componentwise internal scaling during element assembly, constructs the Fourier Green operator, solves the scaled system by matrix-free preconditioned linear CG, and extracts the effective axial stress. This is the final step in Studio.




Inputs

------

n_voxels : int

radius : float

center : array-like of shape (3,)

matrix_lame : array-like of shape (2,)

inclusion_lame : array-like of shape (2,)

tol : float

max_iter : int




Returns

-------

result : dict

    Keys mesh, quadrature, first_cut_element, first_cut_enrichment, elements, green, solution, effective, and final_answer, with nested schemas specified below.

```python
import numpy as np


def run_xfft_pipeline(
    n_voxels: int = 3,
    radius: float = 0.31,
    center: tuple = (0.43, 0.37, 0.52),
    matrix_lame: tuple = (3.0, 2.0),
    inclusion_lame: tuple = (30.0, 20.0),
    tol: float = 1e-10,
    max_iter: int = 500,
) -> dict:
    """Run the full X-FFT toy homogenization and return its numeric state.

    Parameters
    ----------
    n_voxels : int
        Number of periodic voxels per cell edge.
    radius : float
        Radius of the periodic spherical inclusion.
    center : tuple
        Three coordinates of the inclusion center.
    matrix_lame : tuple
        Lamé pair (lambda, mu) of the matrix.
    inclusion_lame : tuple
        Lamé pair (lambda, mu) of the inclusion.
    tol : float
        Relative preconditioned residual tolerance.
    max_iter : int
        Maximum number of linear CG iterations.

    Returns
    -------
    result : dict
        Keys mesh, quadrature, first_cut_element, first_cut_enrichment, elements, green, solution, effective, and final_answer, with nested schemas specified below.
    Raises
    ------
    ValueError
        Propagated from the earlier steps for invalid inputs, in particular
        n_voxels below two, radius outside (0, 0.5), or a center on or outside
        the open unit cell.
    """
    return
```
