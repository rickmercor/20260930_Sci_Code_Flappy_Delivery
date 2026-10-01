# Mathematics-Computational_Mechanics-53

## Background

Laminated composite plates combine layers with different material orientations in order to tailor stiffness, strength, and thermal response. Their effective structural behaviour depends on the through-thickness arrangement of the individual orthotropic plies and includes extensional, bending, transverse-shear, and thermally induced effects.

First-order shear-deformation plate theories account for transverse shear deformation and can therefore model both thin and moderately thick plates. Low-order finite-element discretizations of such theories may, however, become excessively stiff in the thin-plate limit because of shear locking.

Geometrically nonlinear plate analysis introduces additional coupling between transverse deflection and in-plane deformation. Under von Kármán kinematics, gradients of transverse displacement contribute quadratically to the membrane strains. Thermal expansion can additionally generate membrane pre-stress that modifies the tangent structural response.

A robust finite-element treatment of laminated plates therefore requires compatible laminate constitutive data, bending and shear interpolation, nonlinear strain evaluation, numerical integration, and a consistent tangent linearization.

## Problem

Consider the nonlinear four-node laminated-plate element introduced in the cited paper.

Use one rectangular element with dimensions

\[
a = 2.4\ {\rm mm},
\qquad
b = 1.6\ {\rm mm},
\]

centered at the origin.

Use the node ordering

\[
1=(-a/2,-b/2),
\qquad
2=(a/2,-b/2),
\qquad
3=(a/2,b/2),
\qquad
4=(-a/2,b/2).
\]

The laminate consists of four equal-thickness orthotropic plies arranged from the bottom surface to the top surface as

\[
[30^\circ,-30^\circ,-30^\circ,30^\circ].
\]

Each ply has thickness

\[
t_p = 0.2\ {\rm mm}.
\]

Use the temperature-independent ply properties

\[
E_1 = 135000\ {\rm N/mm^2},
\]

\[
E_2 = 9000\ {\rm N/mm^2},
\]

\[
\nu_{12}=0.28,
\]

\[
G_{12}=5000\ {\rm N/mm^2},
\]

\[
G_{13}=4500\ {\rm N/mm^2},
\]

\[
G_{23}=3500\ {\rm N/mm^2}.
\]

The material-axis coefficients of thermal expansion are

\[
\alpha_1=-0.2\times10^{-6}\ {\rm ^\circ C^{-1}},
\]

\[
\alpha_2=28\times10^{-6}\ {\rm ^\circ C^{-1}}.
\]

Use

\[
k_s=\frac56
\]

for the transverse-shear correction factor and apply the uniform temperature change

\[
\Delta T=45^\circ {\rm C}.
\]

Evaluate the prescribed current state using the nonlinear Q4BW formulation defined in the cited paper.

Use the element generalized displacement ordering

\[
\mathbf q=
[
w_1,\phi_{y1},\phi_{x1},
w_2,\phi_{y2},\phi_{x2},
w_3,\phi_{y3},\phi_{x3},
w_4,\phi_{y4},\phi_{x4},
u_1,v_1,u_2,v_2,u_3,v_3,u_4,v_4
]^T.
\]

The transverse and in-plane displacement entries are in mm and the rotational entries are dimensionless.

At the current state,

\[
\mathbf q=
\begin{bmatrix}
0.018\\
0.010\\
-0.006\\
0.027\\
0.006\\
-0.004\\
0.041\\
-0.003\\
0.009\\
0.022\\
-0.005\\
0.007\\
0.0015\\
-0.0008\\
0.0024\\
-0.0011\\
0.0032\\
0.0017\\
0.0011\\
0.0013
\end{bmatrix}.
\]

Construct the consistent tangent of the Q4BW element about this prescribed current state.

For the incremental perturbation, impose zero increments at the following zero-based generalized degrees of freedom:

\[
[0,1,2,3,9,12,13,15].
\]

Apply the generalized incremental load vector \(\Delta\mathbf p\) whose only nonzero entries are

\[
\Delta p_6=-8.0,
\]

\[
\Delta p_8=1.8,
\]

\[
\Delta p_{10}=-2.3,
\]

\[
\Delta p_{14}=3.1,
\]

\[
\Delta p_{17}=-1.7,
\]

\[
\Delta p_{19}=2.4.
\]

Entries conjugate to translational degrees of freedom are forces in N, while entries conjugate to rotational degrees of freedom are moments in N mm.

Solve the constrained linearized incremental problem about the prescribed current state.

Determine the signed transverse displacement increment of node 3,

\[
\Delta w_3.
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

01_transform_orthotropic_ply.py

Goal
----
Transform the reduced orthotropic stiffness and thermal-expansion data of one lamina from its material axes to the global plate axes.

Use the engineering-shear convention for the in-plane strain vector and preserve the sign of the angle-dependent coupling terms.

Return the transformed in-plane reduced stiffness, transformed transverse-shear stiffness, and transformed thermal-expansion vector.

Angles are supplied in degrees. Do not round intermediate values.

```python
import numpy as np


def transform_orthotropic_ply(
    theta_deg: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transformed in-plane stiffness, shear stiffness, and CTE."""

    return (
        np.zeros((3, 3), dtype=float),
        np.zeros((2, 2), dtype=float),
        np.zeros(3, dtype=float),
    )
```

### Step 2

02_build_symmetric_laminate.py

Goal
----
Construct the laminate stiffness and thermal-resultant data for an equal-thickness symmetric stacking sequence.

Transform each orthotropic ply to the global plate axes using the preceding step and integrate its contribution through the laminate thickness.

Return the extensional, extension-bending, bending, and transverse-shear stiffness matrices together with the thermal membrane and thermal moment resultants.

The stacking sequence is supplied from the bottom surface to the top surface. Do not round intermediate values.

```python
import numpy as np


def build_symmetric_laminate(
    angles_deg: np.ndarray,
    ply_thickness: float,
    E1: float,
    E2: float,
    nu12: float,
    G12: float,
    G13: float,
    G23: float,
    alpha1: float,
    alpha2: float,
    delta_T: float,
    shear_correction: float = 5.0 / 6.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return A, B, D, As, NT, and MT."""

    return (
        np.zeros((3, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros((3, 3), dtype=float),
        np.zeros((2, 2), dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
    )
```

### Step 3

03_bw_basis_derivatives.py

Goal
----
Evaluate the transverse-deflection polynomial basis used by the cited Bergan-Wang element together with the derivatives required by the equilibrium-reduced interpolation.



Return one array whose rows are ordered as



H,

H_x,

H_y,

H_xx,

H_xy,

H_yy,

H_xxx,

H_xxy,

H_xyy,

H_yyy,

H_xxxx,

H_xxxy,

H_xxyy,

H_xyyy,

H_yyyy.



Each row must preserve the source ordering of the twelve polynomial coefficients. Evaluate all derivatives analytically.

```python
import numpy as np


def bw_basis_derivatives(
    x: float,
    y: float,
) -> np.ndarray:
    """Return the ordered Bergan-Wang polynomial derivative table."""

    return np.zeros(
        (15, 12),
        dtype=float,
    )
```

### Step 4

04_q4bw_bending_shapes.py

Goal
----
Construct the source Q4BW bending interpolation at one point of a rectangular element centered at the origin.



Use the preceding polynomial derivative table, the supplied laminate bending and transverse-shear stiffness matrices, and the equilibrium-reduced rotation interpolation of the cited method.

Build the twelve-by-twelve nodal basis using the four element nodes in the prescribed counter-clockwise order.

Return the transverse displacement interpolation, its x and y derivatives, the two rotation interpolations, the curvature operator, and the transverse-shear operator.

Use linear solves for the nodal interpolation transformation. Do not round intermediate values.

```python
import numpy as np


def q4bw_bending_shapes(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    x: float,
    y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return Q4BW displacement, rotation, curvature, and shear operators."""

    return (
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros(12, dtype=float),
        np.zeros((3, 12), dtype=float),
        np.zeros((2, 12), dtype=float),
    )
```

### Step 5

05_q4bw_point_kinematics.py

Goal
----
Evaluate the nonlinear plate kinematics at one point of the Q4BW element.



Use the source bending interpolation from the preceding step and standard bilinear interpolation for the in-plane nodal displacements.

The generalized displacement vector is ordered as twelve bending parameters followed by the eight node-wise in-plane parameters.

Evaluate the von Kármán membrane strain, bending curvature, and transverse-shear strain together with their first-variation operators with respect to the complete twenty-component generalized displacement vector.

Do not round intermediate quantities.

```python
import numpy as np


def q4bw_point_kinematics(
    a: float,
    b: float,
    D: np.ndarray,
    As: np.ndarray,
    q: np.ndarray,
    x: float,
    y: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return strains and first-variation operators."""

    return (
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(2, dtype=float),
        np.zeros((3, 20), dtype=float),
        np.zeros((3, 20), dtype=float),
        np.zeros((2, 20), dtype=float),
    )
```

### Step 6

06_assemble_q4bw_tangent.py

Goal
----
Assemble the generalized internal residual and exact current-state tangent of one symmetric-laminate Q4BW element.

Use the source numerical integration rule and the nonlinear strain operators from the preceding step.

Include the extensional, bending, transverse-shear, thermal, and von Kármán geometric contributions consistently.

The tangent must be the analytical first derivative of the generalized internal residual with respect to the twenty generalized element variables.

Return the generalized residual, tangent, and current internal potential. Do not estimate the tangent by finite differences.

```python
import numpy as np


def assemble_q4bw_tangent(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """Return internal residual, consistent tangent, and internal potential."""

    return (
        np.zeros(20, dtype=float),
        np.zeros((20, 20), dtype=float),
        0.0,
    )
```

### Step 7

07_solve_q4bw_increment.py

Goal
----
Compute the constrained generalized displacement increment produced by an incremental generalized load about a prescribed nonlinear Q4BW state.

Construct the consistent tangent at the supplied current state using the preceding step.

Impose zero increments at the supplied constrained zero-based generalized coordinates, solve the reduced tangent system for the remaining free increments, and restore the complete twenty-component incremental vector with zeros at all constrained coordinates.

Return the signed increment associated with target_dof.

Use a direct linear solve and do not round intermediate values.

```python
import numpy as np


def solve_q4bw_increment(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
    delta_p: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:
    """Return the signed increment at target_dof."""

    return 0.0
```
