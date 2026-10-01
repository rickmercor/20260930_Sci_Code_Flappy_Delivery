# Mathematics-Computational_Mechanics-41

## Background

Mixed finite-element formulations introduce additional internal fields in order to improve the numerical behavior of standard displacement-based discretizations. In three-dimensional solid mechanics, independent approximations of mechanical fields can reduce locking, improve bending response, and increase robustness on distorted meshes.

Multi-field variational principles provide a systematic framework for constructing such elements. Additional element variables may be introduced locally and subsequently eliminated before assembly of the global structural equations, so that the final global problem can retain only nodal displacement degrees of freedom.

Quadratic tetrahedral elements use ten displacement nodes and permit curved isoparametric geometry. Under finite deformation, their kinematic operators depend on the current configuration, and consistent linearization produces both constitutive and stress-dependent contributions to the tangent response.

A reliable nonlinear solid-element calculation therefore requires compatible geometric mapping, finite-strain kinematics, constitutive response, numerical integration, local internal-variable treatment, and tangent linearization.

## Problem

Consider the nonlinear 10-node mixed-hybrid tetrahedral element introduced in the cited paper.

Use the following reference natural-coordinate ordering for the ten displacement nodes:

\[
\begin{aligned}
1&:(0,0,0),&
2&:(1,0,0),&
3&:(0,1,0),&
4&:(0,0,1),\\
5&:(1/2,0,0),&
6&:(1/2,1/2,0),&
7&:(0,1/2,0),\\
8&:(0,0,1/2),&
9&:(1/2,0,1/2),&
10&:(0,1/2,1/2).
\end{aligned}
\]

The reference nodal coordinates, in mm, are

\[
\mathbf X =
\begin{bmatrix}
 0.00 & 0.00 & 0.00\\
 1.20 & 0.10 & 0.00\\
 0.10 & 1.10 & 0.05\\
 0.00 & 0.15 & 0.95\\
 0.62 & 0.02 & 0.03\\
 0.68 & 0.62 &-0.02\\
 0.02 & 0.58 & 0.04\\
-0.03 & 0.08 & 0.50\\
 0.58 & 0.15 & 0.50\\
 0.06 & 0.62 & 0.52
\end{bmatrix}.
\]

With \(X,Y,Z\) inserted as their numerical values in mm, prescribe the displacement components in mm by

\[
u_x
=
0.08X
+
0.025Y
+
0.015XZ,
\]

\[
u_y
=
-0.035Y
+
0.020Z
+
0.010XY,
\]

\[
u_z
=
0.060Z
-
0.018X
+
0.012YZ.
\]

Evaluate this state using the nonlinear HWT10 element formulation defined in the cited paper.

For the constitutive response, use the hyperelastic material law employed in the nonlinear patch test of the cited paper with

\[
E = 1000\ {\rm N/mm^2},
\qquad
\nu = 0.3.
\]

Do not substitute a standard displacement-based T10 element for the cited HWT10 formulation.

Construct the condensed element tangent at the prescribed current state.

Constrain the zero-based displacement degrees of freedom

\[
[0,1,2,4,5,8].
\]

The 30 displacement degrees of freedom are ordered as

\[
[u_{1x},u_{1y},u_{1z},
u_{2x},u_{2y},u_{2z},
\dots,
u_{10x},u_{10y},u_{10z}].
\]

Apply the incremental nodal load vector \(\Delta\mathbf p\), in N, whose only nonzero entries are

\[
\Delta p_3=3,
\qquad
\Delta p_7=-2,
\qquad
\Delta p_{11}=-12,
\qquad
\Delta p_{29}=-5.
\]

Solve the constrained linearized incremental problem using the condensed tangent evaluated at the prescribed state.

Determine the signed incremental \(z\)-displacement of node 10,

\[
\Delta u_{10z}.
\]

Use no intermediate rounding.

Report the final value in mm rounded to exactly 12 digits after the decimal point.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

01_tet10_shape_data.py

Goal
----
Evaluate the standard quadratic 10-node tetrahedral displacement interpolation for the task's prescribed natural-coordinate node ordering.

Use the four barycentric coordinates associated with the tetrahedron vertices and preserve the node order specified in the task. Return both the ten shape-function values and their derivatives with respect to xi, eta, and zeta.

The shape functions must satisfy partition of unity, and the sum of their natural derivatives must vanish. Do not round intermediate values.

```python
import numpy as np


def tet10_shape_data(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return T10 shape functions and natural derivatives."""

    return (
        np.zeros(10, dtype=float),
        np.zeros((10, 3), dtype=float),
    )
```

### Step 2

02_tet10_kinematics.py

Goal
----
Evaluate the finite-deformation kinematics of a 10-node isoparametric tetrahedron at one natural-coordinate point.



Use the reference nodal coordinates to form the geometric Jacobian and transform the natural shape-function derivatives to derivatives with respect to the reference Cartesian coordinates.



Using the current nodal positions X + u, construct the deformation gradient, the engineering-Voigt Green-Lagrange strain vector, and the corresponding nonlinear strain-displacement matrix.



Return the reference Jacobian determinant, reference-coordinate shape gradients, deformation gradient, strain vector, and nonlinear B matrix.



Reject an integration point with a nonpositive reference Jacobian determinant. Do not round intermediate values.

```python
import numpy as np


def tet10_kinematics(
    coords: np.ndarray,
    disp: np.ndarray,
    xi: np.ndarray,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return finite-deformation T10 kinematic quantities."""

    return (
        0.0,
        np.zeros((10, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros(6, dtype=float),
        np.zeros((6, 30), dtype=float),
    )
```

### Step 3

03_neo_hooke_green_response.py

Goal
----
Evaluate the compressible Neo-Hookean constitutive model used in the nonlinear patch-test discussion of the cited paper.



The strain input uses engineering Voigt order



[E11, E22, E33, 2E12, 2E13, 2E23].



Construct the right Cauchy-Green tensor from the Green-Lagrange strain and evaluate the source strain-energy density, Second Piola-Kirchhoff stress, and exact consistent material tangent.



The returned 6x6 tangent must map an increment of the engineering Green-Lagrange strain vector to the corresponding increment of the stress vector



[S11, S22, S33, S12, S13, S23].

Do not approximate the tangent by finite differences.

```python
import numpy as np


def neo_hooke_green_response(
    E_green: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """Return energy, Second Piola stress, and consistent tangent."""

    return (
        0.0,
        np.zeros(6, dtype=float),
        np.zeros((6, 6), dtype=float),
    )
```

### Step 4

04_hwt10_mixed_interpolation.py

Goal
----
Construct the independent stress and strain interpolation matrices of the cited HWT10 element at one natural-coordinate point.



Follow the exact parameterization used by the paper, including the constant and varying parts of each six-component field.



Preserve the source ordering of the 24 parameters. Return the complete stress and strain interpolation matrices without rounding intermediate values.

```python
import numpy as np


def hwt10_mixed_interpolation(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HWT10 stress and strain interpolation matrices."""

    return (
        np.zeros((6, 24), dtype=float),
        np.zeros((6, 24), dtype=float),
    )
```

### Step 5

05_project_hwt10_internal_fields.py

Goal
----
Determine the independent strain and stress parameter vectors associated with a prescribed HWT10 displacement state.



Use the source-defined element integration rule, mixed interpolation, constitutive response, and local stationarity equations of the cited formulation.



The returned internal fields must satisfy both local Hu-Washizu stationarity conditions consistently at the prescribed displacement state.



Return the source coupling matrix together with the complete independent strain and stress parameter vectors. Do not round intermediate values.

```python
import numpy as np


def project_hwt10_internal_fields(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return L, independent strain parameters, and stress parameters."""

    return (
        np.zeros((24, 24), dtype=float),
        np.zeros(24, dtype=float),
        np.zeros(24, dtype=float),
    )
```

### Step 6

06_assemble_hwt10_condensed_tangent.py

Goal
----
Assemble the nonlinear condensed tangent stiffness of the cited HWT10 element at a prescribed displacement state.



Use the previously determined independent strain and stress fields together with the source-defined element equations and integration rule.



Construct the local mixed-field tangent consistently and eliminate the element-internal variables according to the cited formulation.



Return the complete condensed displacement tangent together with its material-condensation and geometric contributions. Do not round intermediate values.

```python
import numpy as np


def assemble_hwt10_condensed_tangent(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return condensed, material, and geometric HWT10 tangents."""

    return (
        np.zeros((30, 30), dtype=float),
        np.zeros((30, 30), dtype=float),
        np.zeros((30, 30), dtype=float),
    )
```

### Step 7

07_solve_hwt10_increment.py

Goal
----
Compute a constrained linearized incremental displacement using the condensed nonlinear HWT10 tangent.



Construct the tangent at the supplied current state using the preceding HWT10 steps. Remove the prescribed displacement degrees of freedom, solve the reduced tangent system for the free displacement increment, and restore the full 30-component displacement vector with zero increments at the constrained coordinates.



Return the signed displacement increment associated with target_dof.



Use a direct linear solve. Do not replace the HWT10 tangent by a standard displacement-element stiffness or by a pseudoinverse.

```python
import numpy as np


def solve_hwt10_increment(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
    load: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:
    """Return the signed tangent displacement increment at target_dof."""

    return 0.0
```
