# Mathematics-Numerical_Linear_Algebra-18

## Background

The undamped free vibration of a discretised elastic structure is governed by the generalised eigenvalue problem K phi = lambda M phi, where K and M are the global stiffness and mass matrices assembled from the finite element discretisation, lambda is the square of the circular natural frequency, and phi is the corresponding mode shape. For finite element models of industrial size the direct solution of this problem is expensive enough that it cannot be repeated as often as design iteration, parameter variation or model updating requires, which is the standing motivation for reduced order modelling: one seeks a subspace of modest dimension whose Ritz projection reproduces the lowest part of the spectrum to acceptable accuracy at a small fraction of the cost.

Component mode synthesis is the family of reduction methods that builds such a subspace by partitioning the structure into substructures, representing each independently, and assembling a reduced model for the whole. Its appeal is that the substructures can be reduced separately and that the physical partition of the structure survives into the reduced model, so components can be replaced or re-analysed without rebuilding everything. The classical member of the family represents each substructure by two sets of vectors. The constraint modes are the static deformations of the interior degrees of freedom produced by imposing unit displacement at each interface degree of freedom in turn, with the remaining interface degrees of freedom held fixed. The fixed-interface normal modes are the eigenvectors of the substructure obtained with the entire interface clamped, retained in ascending order of eigenvalue and truncated after a chosen count.

The accuracy of the resulting reduced model is governed by that truncation. The constraint modes represent the static contribution of the interface exactly, so no error is committed there, but discarding the higher fixed-interface normal modes removes their inertial contribution, and the resulting eigenvalue error grows with mode number. Because the reduction is a Ritz projection onto a subspace of the full space, the reduced eigenvalues are bounded below by the exact ones and approach them from above as the subspace is enlarged. A range of schemes has been developed to recover part of the discarded contribution by augmenting the basis with further vectors built from the truncated modes.

Assessing the accuracy actually achieved presents a difficulty of its own. Estimators that evaluate the residual of the reduced solution against the full-order operators require the full-order eigensolution, which defeats the purpose of reducing the model in the first place. An estimator that avoids the full-order eigensolution and works only from reduced quantities and the already-assembled full-order matrices is therefore of practical interest, and is the setting for the problem posed here.

## Problem

A rectangular plate of length 1.0 m, width 0.6 m and thickness 0.01 m is made of an isotropic material with Young's modulus 210 GPa, Poisson's ratio 0.3 and density 7850 kg/m^3. Discretise the plate with a uniform 24 by 12 by 1 grid of eight-node hexahedral elements with trilinear shape functions, evaluating both the stiffness and the consistent mass matrix by 2 by 2 by 2 Gauss quadrature. Every degree of freedom on the face x = 0 is fully restrained; all remaining translational degrees of freedom are free.

Partition the structure into three substructures by the two planes x = L/3 and x = 2L/3. The nodes lying on those two planes constitute the interface; the nodes strictly between the restrained face and x = L/3, strictly between the two planes, and strictly beyond x = 2L/3 constitute the interiors of the first, second and third substructure respectively.

Reduce the structure by component mode synthesis. For each substructure retain, in ascending order of the fixed-interface eigenvalue, ten normal modes for the first substructure, ten for the second and eight for the third, and combine them with the constraint modes describing the static response of the interior to unit interface displacement. Augment that basis with the first-order residual modes obtained from the leading term of the Neumann expansion of the dynamic residual flexibility acting on the coupled inertia force at the interface, normalising the residual mode columns to unit Euclidean length. Project the augmented system back to the dimension of the unaugmented basis by retaining its leading eigenvectors, and solve the resulting reduced eigenvalue problem.

Consider the twentieth mode of that reduced system in ascending order of eigenvalue. Expand it to physical coordinates and separate it into the contribution carried by the unaugmented basis and the contribution carried by the residual-mode columns. Compute the Rayleigh quotient of the first of those two contributions alone and express it as a relative deviation from the reduced eigenvalue of the twentieth mode, forming that relative deviation directly from the two quadratic forms of the base contribution rather than from any simplification of its mass normalisation.

Your final answer must be a single number: that relative deviation, dimensionless.

Report the number of free equations and the size of each of the four blocks of the partitioned ordering, the lowest fixed-interface frequency in hertz of each substructure, the number of residual-mode columns that vanish identically in each substructure, the dimension of the unaugmented basis and of the augmented system before projection, the eigenvalue and the corresponding frequency in hertz of the twentieth reduced mode, and both quadratic forms of the base contribution.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> under 1500 words. Show the intermediate quantities that justify the final number, covering the discretisation, the partition, the construction of the augmented basis, the secondary projection, and the separation of the expanded mode.

Do not paste matrices, coordinate lists, per-element tables, or full eigenvalue spectra.

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

hex8_element_matrices

Goal
----
Form the stiffness and consistent mass matrices of a single eight-node hexahedral element of an isotropic linearly elastic solid.

```python
def hex8_element_matrices(node_coords: 'np.ndarray', E: float, nu: float,
                          rho: float) -> 'np.ndarray':
    '''Stiffness and consistent mass matrix of one eight-node hexahedral element.

    Parameters
    ----------
    node_coords : np.ndarray
        (8, 3) float array of nodal coordinates in the element's own ordering.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    rho : float
        Mass density.

    Returns
    -------
    np.ndarray
        (2, 24, 24) float array; plane 0 is stiffness, plane 1 is consistent mass.
    '''
    return element_matrices  # placeholder
```

### Step 2

assemble_partitioned_system

Goal
----
Assemble the global stiffness and mass matrices of a rectangular box, apply the restraint, and return them reordered into the block structure used by every later step.

```python
def assemble_partitioned_system(nx: int, ny: int, nz: int, Lx: float, Ly: float,
                                Lz: float, E: float, nu: float,
                                rho: float) -> 'np.ndarray':
    '''Assemble the restrained system in partitioned order.

    Parameters
    ----------
    nx, ny, nz : int
        Element counts along the three coordinate directions.
    Lx, Ly, Lz : float
        Edge lengths of the box.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density.

    Returns
    -------
    np.ndarray
        (2, n, n) float array; plane 0 is stiffness, plane 1 is mass, both in the
        four-block partitioned ordering.
    '''
    return partitioned_system  # placeholder
```

### Step 3

fixed_interface_modes

Goal
----
Compute the retained normal modes of substructure k with its entire interface held fixed.

```python
def fixed_interface_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray',
                          k: int, nd: int) -> 'np.ndarray':
    '''Retained fixed-interface normal modes of one substructure.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    counts : np.ndarray
        (4,) integer block sizes of the partitioned ordering.
    k : int
        Substructure index, 0, 1 or 2.
    nd : int
        Number of modes to retain.

    Returns
    -------
    np.ndarray
        (n, nd) float array, zero outside the rows of substructure k.
    '''
    return retained_modes  # placeholder
```

### Step 4

constraint_modes

Goal
----
Compute the static response of the interior of substructure k to unit displacement at each interface degree of freedom in turn, the remaining interface degrees of freedom being held at zero.

```python
def constraint_modes(Kp: 'np.ndarray', counts: 'np.ndarray', k: int) -> 'np.ndarray':
    '''Static interior response to unit interface displacement.

    Parameters
    ----------
    Kp : np.ndarray
        (n, n) partitioned stiffness matrix.
    counts : np.ndarray
        (4,) integer block sizes of the partitioned ordering.
    k : int
        Substructure index, 0, 1 or 2.

    Returns
    -------
    np.ndarray
        (n, nb) float array with nb the interface size, zero outside the interior
        rows of substructure k and zero on the interface rows.
    '''
    return static_response  # placeholder
```

### Step 5

residual_modes

Goal
----
Compute the first-order residual modes of substructure k and scale each column to unit Euclidean length.

```python
def residual_modes(Kp: 'np.ndarray', Mp: 'np.ndarray', counts: 'np.ndarray', k: int,
                   Phi: 'np.ndarray') -> 'np.ndarray':
    '''First-order residual modes of one substructure.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    counts : np.ndarray
        (4,) integer block sizes of the partitioned ordering.
    k : int
        Substructure index, 0, 1 or 2.
    Phi : np.ndarray
        (n, nd) retained-mode array for the same substructure.

    Returns
    -------
    np.ndarray
        (n, nb) float array, zero outside the interior rows of substructure k.
    '''
    return residual_modes_array  # placeholder
```

### Step 6

reduced_eigenvalues

Goal
----
Compute the eigenvalues of the reduced system obtained by projecting the augmented system back to the dimension of its unaugmented part.

```python
def reduced_eigenvalues(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray',
                        nCB: int) -> 'np.ndarray':
    '''Eigenvalues of the secondary-projected reduced system.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    T : np.ndarray
        (n, m) augmented transformation.
    nCB : int
        Number of leading columns of T forming its unaugmented part.

    Returns
    -------
    np.ndarray
        One-dimensional float array of retained eigenvalues, ascending.
    '''
    return eigenvalues  # placeholder
```

### Step 7

base_contribution

Goal
----
Extract the part of one reduced mode that is carried by the unaugmented columns of the basis.

```python
def base_contribution(Kp: 'np.ndarray', Mp: 'np.ndarray', T: 'np.ndarray', nCB: int,
                      mode_index: int) -> 'np.ndarray':
    '''Unaugmented part of one expanded reduced mode.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    T : np.ndarray
        (n, m) augmented transformation.
    nCB : int
        Number of leading columns of T forming its unaugmented part.
    mode_index : int
        One-based index into the ascending retained eigenvalues.

    Returns
    -------
    np.ndarray
        (n,) float array.
    '''
    return base_vector  # placeholder
```

### Step 8

relative_deviation

Goal
----
Express the Rayleigh quotient of a trial vector as a relative deviation from a reference eigenvalue.

```python
def relative_deviation(Kp: 'np.ndarray', Mp: 'np.ndarray', u0: 'np.ndarray',
                       lam: float) -> float:
    '''Rayleigh quotient of u0 as a relative deviation from lam.

    Parameters
    ----------
    Kp, Mp : np.ndarray
        (n, n) partitioned stiffness and mass matrices.
    u0 : np.ndarray
        (n,) trial vector.
    lam : float
        Reference eigenvalue, strictly positive.

    Returns
    -------
    float
    '''
    return 0.0
```

### Step 9

reduction_report

Goal
----
Run the whole pipeline for one configuration and return the quantities the problem statement asks to be reported.

```python
def reduction_report(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float,
                     E: float, nu: float, rho: float, nd: tuple,
                     mode_index: int) -> 'np.ndarray':
    '''Run the whole pipeline and return the reported quantities.

    Parameters
    ----------
    nx, ny, nz : int
        Element counts along the three coordinate directions.
    Lx, Ly, Lz : float
        Edge lengths of the box.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density.
    nd : sequence of int
        Three retained-mode counts, one per substructure.
    mode_index : int
        One-based index into the ascending retained eigenvalues.

    Returns
    -------
    np.ndarray
        (9,) float array as described in the step description.
    '''
    return reported_quantities  # placeholder
```
