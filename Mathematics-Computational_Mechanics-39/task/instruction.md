# Mathematics-Computational_Mechanics-39

## Background

Coupled computational methods can exploit different discretization techniques in different regions of a mechanical system. Mesh-based finite elements are efficient and accurate when deformation remains moderate, whereas particle-based discretizations can remain robust when local deformation becomes too severe for a conventional mesh.

When two independently discretized subdomains meet at an interface, their displacement fields need not share matching nodes. Interface constraints can therefore be imposed weakly rather than by direct node-to-node identification.

Mortar methods provide such a weak coupling framework by introducing an interface interpolation for the constraint field and integrating coupling quantities over the interface. Suitable dual interpolation functions can simplify the resulting algebra and enable elimination of interface unknowns.

A robust implementation therefore requires interface interpolation, numerical integration across nonmatching discretizations, construction of coupling operators, condensation of constrained degrees of freedom, and solution of the reduced mechanical system.

## Problem

Consider one scalar displacement component of the coupled implicit MPM-FEM dual-mortar formulation introduced in the cited paper.

The coupling interface is a rectangular surface centered at the origin with dimensions

$$
L_x=2.4\ {\rm mm},
\qquad
L_y=1.6\ {\rm mm}.
$$

The non-mortar side is represented by one bilinear quadrilateral finite-element surface with the node ordering

$$
1=(-L_x/2,-L_y/2),
$$

$$
2=(L_x/2,-L_y/2),
$$

$$
3=(L_x/2,L_y/2),
$$

$$
4=(-L_x/2,L_y/2).
$$

Use the natural-coordinate domain

$$
(\xi,\eta)\in[-1,1]\times[-1,1].
$$

For this benchmark, specialize the mortar-side MPM background-grid trial interpolation to standard bilinear functions on a regular Cartesian grid covering the same physical interface. The dual-mortar coupling formulation itself is to be used as defined in the cited paper.

Use

$$
n_x=3
$$

cells in the x direction and

$$
n_y=2
$$

cells in the y direction.

Thus the mortar-side background grid contains 12 displacement degrees of freedom. Order them row-major, from the bottom row to the top row and from left to right within each row.

Use the dual-mortar interface formulation, interface quadrature prescription, dual interpolation functions, mortar projection, and static-condensation procedure defined in the cited paper.

The interface does not intersect an additional Dirichlet boundary, so no boundary modification of the dual multiplier basis is required.

At the current Newton iteration, all essential boundary conditions outside the coupling interface have already been incorporated into the following uncoupled tangent and residual blocks.

The uninvolved degrees of freedom are

$$
\Delta u^{(n)}\in\mathbb{R}^{3},
$$

the non-mortar finite-element interface degrees of freedom are

$$
\Delta u^{(1)}\in\mathbb{R}^{4},
$$

and the mortar-side background-grid degrees of freedom are

$$
\Delta u^{(2)}\in\mathbb{R}^{12}.
$$

Use

$$
K^{(n,n)}=
\begin{bmatrix}
22 & -3 & 1.5\\
-3 & 20 & -2\\
1.5 & -2 & 18
\end{bmatrix},
$$

$$
K^{(1,1)}=
\begin{bmatrix}
18 & -2 & 0.5 & -1\\
-2 & 17 & -1 & 0.4\\
0.5 & -1 & 16 & -1.5\\
-1 & 0.4 & -1.5 & 15
\end{bmatrix},
$$

and

$$
K^{(2,2)}=
\begin{bmatrix}
21.5&-2&0&0&-1.5&0&0&0&0&0&0&0\\
-2&23.5&-2&0&0&-1.5&0&0&0&0&0&0\\
0&-2&23.5&-2&0&0&-1.5&0&0&0&0&0\\
0&0&-2&21.5&0&0&0&-1.5&0&0&0&0\\
-1.5&0&0&0&23&-2&0&0&-1.5&0&0&0\\
0&-1.5&0&0&-2&25&-2&0&0&-1.5&0&0\\
0&0&-1.5&0&0&-2&25&-2&0&0&-1.5&0\\
0&0&0&-1.5&0&0&-2&23&0&0&0&-1.5\\
0&0&0&0&-1.5&0&0&0&21.5&-2&0&0\\
0&0&0&0&0&-1.5&0&0&-2&23.5&-2&0\\
0&0&0&0&0&0&-1.5&0&0&-2&23.5&-2\\
0&0&0&0&0&0&0&-1.5&0&0&-2&21.5
\end{bmatrix}.
$$

The coupling blocks between the uninvolved and non-mortar coordinates are

$$
K^{(n,1)}=
\begin{bmatrix}
0.8&-0.4&0.2&0.5\\
-0.3&0.7&-0.5&0.1\\
0.4&0.2&0.6&-0.7
\end{bmatrix},
$$

and between the uninvolved and mortar coordinates

$$
K^{(n,2)}=
\begin{bmatrix}
0.20&-0.10&0.05&0.12&0.08&-0.04&0.03&0.06&0.02&-0.01&0.04&0.05\\
-0.05&0.18&-0.08&0.03&-0.02&0.11&-0.04&0.07&0.01&0.06&-0.03&0.09\\
0.10&0.02&0.15&-0.06&0.05&0.04&0.12&-0.03&0.07&0.01&0.09&-0.02
\end{bmatrix}.
$$

Use symmetry for the transpose blocks.

The uncoupled residual blocks are

$$
R^{(n)}=
\begin{bmatrix}
1.2\\
-0.7\\
0.9
\end{bmatrix}\ {\rm N},
$$

$$
R^{(1)}=
\begin{bmatrix}
0.55\\
-0.35\\
0.25\\
-0.45
\end{bmatrix}\ {\rm N},
$$

and

$$
R^{(2)}=
\begin{bmatrix}
0.15\\
-0.20\\
0.10\\
-0.05\\
0.30\\
-0.25\\
0.12\\
-0.18\\
0.22\\
-0.14\\
0.08\\
-0.11
\end{bmatrix}\ {\rm N}.
$$

All tangent blocks have units N/mm.

Compute the dual-mortar coupling projection using the cited paper, form the statically condensed Newton system, solve for the reduced increment, and reconstruct the eliminated non-mortar finite-element interface increment.

Determine the signed increment at non-mortar interface node 3, i.e. the third component of

$$
\Delta u^{(1)}.
$$

Use no intermediate rounding.

Report the final displacement increment in mm rounded to exactly 12 digits after the decimal point.

Output Format Requirements:
Place the final numerical value in the following tag format and make the final-answer tag the last content in the response:

<final_answer>signed_decimal_value</final_answer>

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

01_dual_mortar_q4_shapes.py

Goal
----
Evaluate the primal and source dual interpolation functions on one bilinear quadrilateral non-mortar interface element.

Use the standard four-node bilinear primal basis in the supplied natural coordinates and construct the four dual multiplier interpolation values exactly as defined for the three-dimensional interface formulation in the cited paper.

Preserve the node ordering 1,2,3,4 corresponding to bottom-left, bottom-right, top-right, top-left.

Return the primal and dual interpolation rows.

```python
import numpy as np


def dual_mortar_q4_shapes(
    xi: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return primal Q4 and source dual interface interpolation rows."""

    return (
        np.zeros(4, dtype=float),
        np.zeros(4, dtype=float),
    )
```

### Step 2

02_coupling_gauss_rule.py

Goal
----
Construct the one-dimensional Gauss-Legendre rule used for integration of the MPM-FEM coupling interface.

Determine the integration order from the supplied finite-element interface side lengths and mortar-side grid spacings using the empirical coupling-integration prescription defined in the cited paper.

Apply the floor operation to the mathematical length ratio. If floating-point evaluation places a value within 1e-12 of an integer, treat it as that integer before applying the floor.

Return the selected order together with the corresponding one-dimensional Gauss points and weights on [-1,1].

```python
import numpy as np


def coupling_gauss_rule(
    fe_lengths: np.ndarray,
    mpm_spacing: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return source coupling Gauss order, points, and weights."""

    return (
        1,
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
    )
```

### Step 3

03_regular_grid_interface_shapes.py

Goal
----
Evaluate the mortar-side background-grid trial interpolation used by this benchmark specialization of the dual-mortar coupling algorithm.

The mortar side is a regular Cartesian MPM background grid covering the rectangular interface. For this benchmark, use standard bilinear background-grid trial functions in the cell containing the supplied physical point.

Order all global background-grid nodes row-major from the bottom row to the top row and from left to right within each row.

Return the complete global interpolation row, with zeros for inactive grid nodes.

```python
import numpy as np


def regular_grid_interface_shapes(
    x: float,
    y: float,
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
) -> np.ndarray:
    """Return global bilinear background-grid interpolation row."""

    return np.zeros(
        (n_cells_x + 1)
        * (n_cells_y + 1),
        dtype=float,
    )
```

### Step 4

04_assemble_dual_mortar_projection.py

Goal
----
Assemble the dual-mortar coupling matrices for one rectangular non-mortar Q4 surface coupled to the supplied regular mortar-side grid.

Use the source interface quadrature order from the preceding step.

At every tensor-product Gauss point evaluate the source primal and dual non-mortar functions, map the point to the physical interface, evaluate the mortar-side trial interpolation, and integrate the two source coupling matrices.

Construct the mortar projection by solving the resulting dual coupling system.

Return D, A, P, and the one-dimensional Gauss order.

```python
import numpy as np


def assemble_dual_mortar_projection(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return D, A, dual-mortar projection P, and Gauss order."""

    n2 = (
        (n_cells_x + 1)
        * (n_cells_y + 1)
    )

    return (
        np.zeros((4, 4), dtype=float),
        np.zeros((4, n2), dtype=float),
        np.zeros((4, n2), dtype=float),
        1,
    )
```

### Step 5

05_condense_dual_mortar_newton.py

Goal
----
Construct the statically condensed Newton system corresponding to the cited dual-mortar elimination procedure.

Use the supplied mortar projection and uncoupled tangent/residual blocks.

Eliminate the non-mortar interface degrees of freedom according to the source relation and retain the uninvolved and mortar-side coordinates.

Return the reduced tangent matrix and reduced residual in that retained ordering.

```python
import numpy as np


def condense_dual_mortar_newton(
    P: np.ndarray,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source statically condensed tangent and residual."""

    n = np.asarray(Knn).shape[0]
    n2 = np.asarray(K22).shape[0]

    return (
        np.zeros(
            (n + n2, n + n2),
            dtype=float,
        ),
        np.zeros(
            n + n2,
            dtype=float,
        ),
    )
```

### Step 6

06_solve_dual_mortar_increment.py

Goal
----
Solve the complete statically condensed dual-mortar Newton increment for the supplied interface geometry and uncoupled tangent/residual blocks.

Construct the source mortar projection, form the source condensed Newton system, solve the retained linear system directly, and reconstruct the eliminated non-mortar interface increment using the mortar projection.

Return the uninvolved increment, reconstructed non-mortar increment, mortar-side increment, and the infinity norm of the reduced Newton residual.

Do not round intermediate values.

```python
import numpy as np


def solve_dual_mortar_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return retained and reconstructed Newton increments."""

    return (
        np.zeros(Knn.shape[0], dtype=float),
        np.zeros(4, dtype=float),
        np.zeros(K22.shape[0], dtype=float),
        0.0,
    )
```

### Step 7

07_dual_mortar_target_increment.py

Goal
----
Compute one requested reconstructed non-mortar interface increment for the complete benchmark dual-mortar problem.

Use the complete source coupling, condensation, reduced solve, and eliminated-interface reconstruction from the preceding steps.

Return the signed component of the reconstructed non-mortar interface increment identified by the supplied zero-based target index.

Do not round intermediate values.

```python
import numpy as np


def dual_mortar_target_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    target_nonmortar_dof: int,
) -> float:
    """Return one reconstructed non-mortar Newton increment."""

    return 0.0
```
