# Mathematics-Computational_Mechanics-9

## Background

Structural topology optimization distributes a fixed amount of material inside a design domain so that the resulting structure is as stiff as possible. When the structure is meant to operate at large strain, as compliant mechanisms and soft actuators do, the analysis rather than the optimizer becomes the bottleneck, because mesh-based solvers lose convergence once elements in the emerging void regions distort or invert.

Work in this area replaces the deforming mesh with a hybrid discretisation in which Lagrangian particles carry the material state and a fixed Eulerian grid carries the force balance. Making that analysis differentiable end to end then supplies the design derivatives on which gradient-based optimization depends.

Such frameworks are judged by whether the forward solver reproduces known large-deflection benchmarks and by whether the computed sensitivities agree with finite-difference estimates.

## Problem

Topology optimization of structures that undergo large deformation is limited less by the optimizer than by the analysis, because in a Lagrangian finite element setting the elements in low-density regions distort, invert and tangle long before the design converges. Discretising the body instead as material points that move through a fixed background grid removes that failure mode, and the analysis then takes a pseudo-density carried on each material point and returns the structural compliance together with its derivative with respect to every pseudo-density.

Equilibrium is quasi-static and is enforced on the grid: particle stresses are mapped to internal nodal forces by material point quadrature of the internal virtual work, using the current particle volumes and the shape function gradients taken with respect to the current configuration, and the resulting nonlinear residual is driven to zero by Newton-Raphson. The particle to grid mapping is a generalized interpolation material point basis, whose support reaches beyond a single background cell because each point carries a domain of finite size rather than acting as a point mass. The constitutive law is Hencky hyperelasticity, written in terms of the logarithmic strain of the left Cauchy-Green tensor and closed under a plane-stress condition, with the material properties interpolated from the pseudo-densities by SIMP with exponent q. The design derivative follows from the implicit function theorem applied to the converged residual, so it needs the consistent tangent of the hyperelastic model rather than a secant or a small-strain approximation.

Solve one deterministic instance of this analysis using the following configuration:

- design domain: the rectangle 0 <= x <= 2.0, 0 <= y <= 1.0, of unit thickness
- material points: a uniform lattice of spacing d = 0.2 filling the design domain, one point at the centre of each lattice cell, giving 50 points of reference volume 0.04 each
- background grid: uniform square cells of size h = 0.5 spanning the same rectangle, giving 15 nodes
- particle domain length: l_p = 0.2 in both directions, held at this reference value throughout
- solid material: Young's modulus E_0 = 1.0, Poisson's ratio nu = 0.3, plane stress
- SIMP exponent: q = 3
- pseudo-densities: gamma_p = 0.3 + |y_p - 0.5|, evaluated at the reference position of material point p
- boundary conditions: every grid node at x = 0 is fixed in both directions
- load: a total force of 0.006 in the negative y direction, shared equally by the five material points at x = 1.9
- load history: the whole load is applied in a single increment, and the particle to grid mapping is formed once from the reference material point positions and held fixed through that increment
- solver: Newton-Raphson started from zero nodal displacement, converged when the Euclidean norm of the residual on the unconstrained degrees of freedom falls below 1e-10, at most 50 iterations

Compute the converged nodal displacement field, the compliance of the loaded structure, and the derivative of the compliance with respect to each of the 50 pseudo-densities. Your final answer must be a single number: the most negative entry of that sensitivity vector, which is the largest compliance reduction per unit increase in pseudo-density available at any single material point.

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

01_initialize_material_points

Goal
----
Populate a rectangular design domain with a uniform lattice of material points

and return their reference coordinates together with their reference volumes.

```python
import numpy as np


def initialize_material_points(domain: np.ndarray, point_spacing: float):
    """Lay out material points on a uniform lattice filling a rectangular domain.

    Parameters
    ----------
    domain : np.ndarray
        Side lengths (L_x, L_y) of the design domain.
    point_spacing : float
        Lattice spacing d between neighbouring material points.

    Returns
    -------
    point_state : tuple of np.ndarray
        The pair (points, volumes), holding the reference coordinates of shape
        (n_points, 2) ordered with the x index varying fastest, and the
        reference volume of each material point, of shape (n_points,).

    Raises
    ------
    ValueError
        If `domain` does not hold two positive side lengths, if `point_spacing`
        is not positive, or if either side length is not an integer multiple of
        `point_spacing`.
    """
    return point_state
```

### Step 2

02_gimp_shape_functions

Goal
----
Evaluate the generalized interpolation material point basis and its spatial

gradient for every material point and grid node pair, returning the weight array

and the gradient array that carry every particle to grid transfer in the solver.

```python
import numpy as np


def gimp_shape_functions(points: np.ndarray, nodes: np.ndarray, cell_size: float,
                         point_domain_length: float):
    """Evaluate the generalized interpolation basis and its gradient.

    Parameters
    ----------
    points : np.ndarray
        Material point coordinates, shape (n_points, 2).
    nodes : np.ndarray
        Grid node coordinates, shape (n_nodes, 2).
    cell_size : float
        Uniform grid cell size h.
    point_domain_length : float
        Particle domain length l_p, the same in both directions.

    Returns
    -------
    basis : tuple of np.ndarray
        The pair (shape_values, shape_gradients), holding the basis weights of
        shape (n_points, n_nodes) and the basis gradients of shape
        (n_points, n_nodes, 2).

    Raises
    ------
    ValueError
        If `points` or `nodes` is not of shape (n, 2), if `cell_size` is not
        positive, or if `point_domain_length` lies outside (0, cell_size].
    """
    return basis
```

### Step 3

03_simp_lame_parameters

Goal
----
Interpolate the Lame parameters carried by each material point from its

pseudo-density using the solid isotropic material with penalization rule, and

return them as one array with a column for lambda and a column for mu.

```python
import numpy as np


def simp_lame_parameters(densities: np.ndarray, youngs_modulus: float,
                         poisson_ratio: float, penalty: float) -> np.ndarray:
    """Interpolate the per-point Lame parameters from the pseudo-densities.

    Parameters
    ----------
    densities : np.ndarray
        Pseudo-densities gamma_p in (0, 1], shape (n_points,).
    youngs_modulus : float
        Young modulus E_0 of the solid phase.
    poisson_ratio : float
        Poisson ratio nu of the solid phase.
    penalty : float
        SIMP exponent q.

    Returns
    -------
    lame : np.ndarray
        Interpolated Lame parameters, shape (n_points, 2), with lambda_p in
        column 0 and mu_p in column 1.

    Raises
    ------
    ValueError
        If `densities` is not a non-empty one-dimensional array, if any density
        lies outside (0, 1], if `youngs_modulus` is not positive, if
        `poisson_ratio` lies outside (-1, 0.5), or if `penalty` is less than 1.
    """
    return lame
```

### Step 4

04_hencky_cauchy_stress

Goal
----
Evaluate the Hencky hyperelastic constitutive law under plane stress at every

material point, returning the Cauchy stress and the Jacobian of the deformation

gradient.

```python
import numpy as np


def hencky_cauchy_stress(deformation_gradients: np.ndarray, lame: np.ndarray):
    """Evaluate the plane-stress Hencky constitutive law at every material point.

    Parameters
    ----------
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    lame : np.ndarray
        Lame parameters, shape (n_points, 2), with lambda_p in column 0 and
        mu_p in column 1.

    Returns
    -------
    stress_state : tuple of np.ndarray
        The pair (cauchy_stress, jacobian), holding the in-plane Cauchy stress
        of shape (n_points, 2, 2) and the determinant of each deformation
        gradient, of shape (n_points,).

    Raises
    ------
    ValueError
        If `deformation_gradients` is not of shape (n_points, 2, 2), if `lame`
        is not of shape (n_points, 2), if any point has mu <= 0 or
        lambda + 2 mu <= 0, or if any deformation gradient has a non-positive
        determinant.
    """
    return stress_state
```

### Step 5

05_assemble_nodal_forces

Goal
----
Transfer the material point state to the background grid, returning the internal

nodal force vector assembled from the particle stresses and the external nodal

force vector assembled from the applied point loads.

```python
import numpy as np


def assemble_nodal_forces(cauchy_stress: np.ndarray, deformation_gradients: np.ndarray,
                          volumes: np.ndarray, shape_values: np.ndarray,
                          shape_gradients: np.ndarray, point_loads: np.ndarray):
    """Assemble the internal and external nodal force vectors.

    Parameters
    ----------
    cauchy_stress : np.ndarray
        Cauchy stress at each material point, shape (n_points, 2, 2).
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_values : np.ndarray
        Basis weights, shape (n_points, n_nodes).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    point_loads : np.ndarray
        External force applied at each material point, shape (n_points, 2).

    Returns
    -------
    nodal_forces : tuple of np.ndarray
        The pair (internal_forces, external_forces), both of shape
        (n_nodes, 2).

    Raises
    ------
    ValueError
        If `shape_values` is not two dimensional, if the stress, deformation
        gradient, volume or point load arrays are not sized by n_points, if
        `shape_gradients` is not of shape (n_points, n_nodes, 2), or if any
        deformation gradient has a non-positive determinant.
    """
    return nodal_forces
```

### Step 6

06_assemble_tangent_stiffness

Goal
----
Assemble the consistent tangent stiffness matrix of the discrete force balance

and impose the Dirichlet constraints on it, returning the matrix used both by

the Newton correction and by the adjoint solve.

```python
import numpy as np


def assemble_tangent_stiffness(deformation_gradients: np.ndarray, lame: np.ndarray,
                               volumes: np.ndarray, shape_gradients: np.ndarray,
                               fixed_nodes: np.ndarray) -> np.ndarray:
    """Assemble the constrained consistent tangent stiffness matrix.

    Parameters
    ----------
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    lame : np.ndarray
        Lame parameters, shape (n_points, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    fixed_nodes : np.ndarray
        Boolean mask of constrained nodes, shape (n_nodes,).

    Returns
    -------
    stiffness : np.ndarray
        Tangent stiffness with constrained rows and columns replaced by the
        identity, shape (2 n_nodes, 2 n_nodes), with degree of freedom
        2 v + i belonging to component i of node v.

    Raises
    ------
    ValueError
        If `shape_gradients` is not of shape (n_points, n_nodes, 2), if
        `deformation_gradients`, `lame` or `volumes` is not sized by n_points, if
        `fixed_nodes` is not of shape (n_nodes,), or if any deformation gradient
        has a non-positive determinant.
    """
    return stiffness
```

### Step 7

07_compliance_and_adjoint

Goal
----
Evaluate the structural compliance of the converged equilibrium state and solve

the adjoint system that the design derivative of that compliance requires.

```python
import numpy as np


def compliance_and_adjoint(stiffness: np.ndarray, external_forces: np.ndarray,
                           displacement: np.ndarray, fixed_nodes: np.ndarray):
    """Evaluate the compliance and solve for its adjoint field.

    Parameters
    ----------
    stiffness : np.ndarray
        Constrained tangent stiffness, shape (2 n_nodes, 2 n_nodes).
    external_forces : np.ndarray
        External nodal forces, shape (n_nodes, 2).
    displacement : np.ndarray
        Converged nodal displacement, shape (n_nodes, 2).
    fixed_nodes : np.ndarray
        Boolean mask of constrained nodes, shape (n_nodes,).

    Returns
    -------
    objective : tuple
        The pair (compliance, adjoint), holding the work done by the external
        forces on the converged displacement as a float and the adjoint field
        of the compliance functional of shape (n_nodes, 2).

    Raises
    ------
    ValueError
        If `external_forces` is not of shape (n_nodes, 2), if `displacement`
        does not have that same shape, if `stiffness` is not of shape
        (2 n_nodes, 2 n_nodes), or if `fixed_nodes` is not of shape (n_nodes,).
    """
    return objective
```

### Step 8

08_density_sensitivity

Goal
----
Contract the adjoint field with the design derivative of the residual to give

the derivative of the compliance with respect to the pseudo-density of every

material point.

```python
import numpy as np


def density_sensitivity(adjoint: np.ndarray, cauchy_stress: np.ndarray,
                        deformation_gradients: np.ndarray, volumes: np.ndarray,
                        shape_gradients: np.ndarray, densities: np.ndarray,
                        penalty: float) -> np.ndarray:
    """Compute the compliance sensitivity with respect to every pseudo-density.

    Parameters
    ----------
    adjoint : np.ndarray
        Adjoint field, shape (n_nodes, 2).
    cauchy_stress : np.ndarray
        Converged Cauchy stress, shape (n_points, 2, 2).
    deformation_gradients : np.ndarray
        Converged deformation gradients, shape (n_points, 2, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    densities : np.ndarray
        Pseudo-densities, shape (n_points,).
    penalty : float
        SIMP exponent q.

    Returns
    -------
    sensitivity : np.ndarray
        Compliance sensitivity, shape (n_points,).

    Raises
    ------
    ValueError
        If `shape_gradients` is not of shape (n_points, n_nodes, 2), if
        `adjoint` is not of shape (n_nodes, 2), if the stress or deformation
        gradient array is not of shape (n_points, 2, 2), if `volumes` or
        `densities` is not of shape (n_points,), if any density is not strictly
        positive, if `penalty` is less than 1, or if any deformation gradient has
        a non-positive determinant.
    """
    return sensitivity
```

### Step 9

09_run_full_pipeline

Goal
----
Chain every earlier stage into the full implicit material point analysis of the

loaded design and return the most negative entry of the compliance sensitivity

vector, the largest compliance reduction available per unit increase in the

pseudo-density of a single material point.

```python
def run_full_pipeline(total_load: float = 0.006, penalty: float = 3.0) -> float:
    '''Run the full analysis and return the most negative compliance sensitivity.

    Parameters
    ----------
    total_load : float
        Total force in the negative y direction shared equally by the material
        points on the loaded edge.
    penalty : float
        SIMP exponent q.

    Returns
    -------
    minimum_sensitivity : float
        The most negative entry of the compliance sensitivity vector.

    Raises
    ------
    ValueError
        If `total_load` or `penalty` is not a real number, if `total_load` is
        not positive, or if `penalty` is less than 1.
    '''
    return minimum_sensitivity
```
