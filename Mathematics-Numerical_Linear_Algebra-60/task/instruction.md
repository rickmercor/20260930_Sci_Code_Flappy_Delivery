# Mathematics-Numerical_Linear_Algebra-60

## Background

Finite element methods approximate a continuous variational problem by restricting its trial and test functions to finite-dimensional spaces defined over a triangulation of the computational domain. The geometry and connectivity of the triangulation determine how local element contributions are connected into a global discrete problem, while refinement provides a systematic way to increase spatial resolution. For mixed finite element methods, the resulting discrete spaces must preserve the structural properties required by the underlying coupled formulation. The Mini element is one such mixed construction, combining low-order conforming approximation with local enrichment to obtain a stable finite-dimensional space.

## Problem

The supplied paper develops a locking-free Mini mixed finite element approach for nearly incompressible linear-elasticity eigenproblems; use that paper as the authoritative definition of the mathematical problem, finite-element spaces, and discrete eigenvalue formulation. Do not use numerical values from the paper's experiments as substitutes for the computation requested here. For the present instance, take $\Omega=(0,1)^2$, homogeneous Dirichlet conditions for displacement, $\mu=1$, and $\lambda=7777$.

Generate the triangulation from a $48\times48$ Cartesian partition of $\Omega$, with cells indexed by integers $0\le i,j<48$; in cell $(i,j)$, select the southwest-to-northeast diagonal exactly when $(i+3j)\bmod 5\in\{0,1\}$, and select the other diagonal otherwise. Apply one uniform refinement to this triangulation before constructing the finite element spaces. Apart from this mesh construction and the stated parameter values, follow the supplied paper's formulation without introducing a different elasticity discretization.

Using the resulting discrete mixed eigenproblem, compute the second-smallest positive discrete eigenvalue, counting multiplicities in the ordering defined by the paper. The numerical value must come from the assembled finite element problem for this instance rather than from the continuous spectrum, an interpolation or asymptotic estimate, or a value copied from the source paper. Use sufficient numerical precision for the reported scalar to be stable under independent implementations of the same mathematical problem.
Report the first four positive discrete eigenvalues in increasing order, retaining multiplicities, and identify the second-smallest positive eigenvalue requested above.
In addition, report the third positive eigenvalue and the spectral gap

$$
\Delta_{23}=\theta_3-\theta_2
$$

between the second- and third-smallest positive discrete eigenvalues, with both quantities obtained from the same assembled discrete mixed finite-element problem.

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

01_build_mesh.py

Goal
----
Construct the deterministic triangulation used by the benchmark. The mesh starts from a uniform Cartesian partition of the unit square. Each cell is split according to the benchmark-specific diagonal rule, followed by one uniform red refinement of every triangle. The implementation is deterministic and returns consistently oriented triangles together with the vertex coordinates.

```python
import numpy as np

def build_mesh(nx: int, ny: int, refine: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the deterministic benchmark triangulation.

    The initial mesh is the Cartesian partition of (0,1)^2 into nx by ny
    square cells. Initial vertices use the deterministic numbering

        vid(i, j) = j * (nx + 1) + i,

    with i increasing from left to right and j increasing from bottom to top.
    Initial square cells are visited in increasing j, then increasing i.

    For cell (i, j), use the SW-NE diagonal when

        (i + 3*j) % 5 in {0, 1},

    and use the opposite diagonal otherwise. All triangles are stored with
    counter-clockwise vertex order.

    Each uniform refinement level is a red refinement: every triangle is
    divided into four child triangles by inserting the three edge midpoints.
    Existing vertex indices are preserved. New midpoint vertices are appended
    in the first-encountered order while traversing the triangles, so the
    refinement is deterministic. The four children of each parent are emitted
    in the fixed local order induced by the parent triangle and its three
    edge midpoints, and each child is stored counter-clockwise.

    Parameters
    ----------
    nx : int
        Number of Cartesian cells in the x direction.
    ny : int
        Number of Cartesian cells in the y direction.
    refine : int
        Number of uniform red-refinement levels.

    Returns
    -------
    vertices : np.ndarray
        Array of shape (N, 2) containing vertex coordinates. Vertices are
        indexed according to the deterministic numbering convention above,
        with newly created refinement midpoints appended in first-encounter
        order.
    triangles : np.ndarray
        Integer array of shape (T, 3) containing counter-clockwise triangle
        vertex indices.

    Raises
    ------
    ValueError
        If nx <= 0, ny <= 0, or refine < 0.
    """
    return vertices, triangles
```

### Step 2

02_build_mini_space.py

Goal
----
Construct the global degree-of-freedom maps for the Mini mixed element. The displacement contains continuous P1 vertex degrees of freedom plus one element-local cubic bubble per triangle for each displacement component. The pressure space contains continuous P1 vertex degrees of freedom.

```python
import numpy as np

def build_mini_space(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build Mini mixed finite-element degree-of-freedom maps.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates of shape (N, 2).
    triangles : np.ndarray
        Triangle indices of shape (T, 3).

    Returns
    -------
    maps : dict
        Deterministic global degree-of-freedom and element-connectivity maps
        with the following required keys:

        - ``vertex_count`` : np.ndarray, shape (1,)
            Number of mesh vertices.
        - ``triangle_count`` : np.ndarray, shape (1,)
            Number of refined mesh triangles.
        - ``scalar_displacement_count`` : np.ndarray, shape (1,)
            Number of enriched scalar displacement degrees of freedom, that
            is the continuous P1 vertex DOFs together with the one bubble
            DOF per triangle.
        - ``displacement_count`` : np.ndarray, shape (1,)
            Total number of displacement degrees of freedom for the two
            components, equal to ``2 * scalar_displacement_count``.
        - ``pressure_count`` : np.ndarray, shape (1,)
            Number of continuous scalar P1 pressure degrees of freedom.
        - ``element_scalar`` : np.ndarray, shape (T, 4)
            Local scalar Mini-element DOF map for each triangle, containing
            the three vertex P1 DOFs followed by the triangle bubble DOF.
        - ``element_x`` : np.ndarray, shape (T, 4)
            Global x-component displacement DOF map corresponding to
            ``element_scalar``.
        - ``element_y`` : np.ndarray, shape (T, 4)
            Global y-component displacement DOF map corresponding to
            ``element_scalar``.
    """
    return maps
```

### Step 3

03_local_matrices.py

Goal
----
Compute the local Mini finite-element matrices on one physical triangle. The local displacement basis consists of the three P1 vertex basis functions and one normalized cubic bubble, for each displacement component. The pressure basis consists of the three P1 vertex functions.

```python
import numpy as np


def local_matrices(
    triangle_vertices: np.ndarray,
    mu: float,
    lam: float,
) -> dict[str, np.ndarray]:
    """
    Assemble the local Mini-element matrices for one triangle.

    The scalar Mini basis is ordered as

        [l1, l2, l3, phi_b],

    where l1, l2, and l3 are the barycentric P1 basis functions associated
    with triangle_vertices[0], triangle_vertices[1], and triangle_vertices[2],
    respectively, and the normalized cubic bubble is

        phi_b = 27 * l1 * l2 * l3.

    The local displacement space has 8 degrees of freedom: the four scalar
    Mini basis functions for the x component followed by the same four basis
    functions for the y component. Thus the local displacement ordering is

        [x_l1, x_l2, x_l3, x_b, y_l1, y_l2, y_l3, y_b].

    The pressure space uses the three scalar P1 basis functions in the
    vertex order [l1, l2, l3].

    Parameters
    ----------
    triangle_vertices : np.ndarray
        Coordinates of one triangle, shape (3, 2). Row 0, row 1, and row 2
        define the vertex order associated with l1, l2, and l3.
    mu : float
        Positive shear modulus.
    lam : float
        Positive Lame parameter.

    Returns
    -------
    matrices : dict[str, np.ndarray]
        Dictionary containing the local matrices for the Mini mixed element.
        The displacement-related matrices use the 8-DOF ordering described
        above, while the pressure-related matrix uses the 3-DOF ordering
        [l1, l2, l3].

        The returned dictionary contains the local displacement stiffness,
        displacement mass, divergence coupling, and pressure matrix.

    Raises
    ------
    ValueError
        If triangle_vertices does not have shape (3, 2), if the triangle is
        degenerate, or if mu <= 0 or lam <= 0.
    """
    return matrices
```

### Step 4

04_assemble_mixed_system.py

Goal
----
Assemble the global Mini mixed finite-element matrices from the mesh and local element operators.

```python
import numpy as np

def assemble_mixed_system(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mu: float,
    lam: float,
) -> dict:
    """
    Assemble the global Mini mixed finite-element matrices.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices.
    triangles : np.ndarray
        Triangle connectivity.
    mu : float
        Shear modulus.
    lam : float
        Lame parameter.

    Returns
    -------
    system : dict
        Global sparse mixed finite-element system containing the following
        required keys:

        - ``A`` : sparse matrix
            Global displacement stiffness matrix for the two-component
            Mini displacement space.
        - ``B`` : sparse matrix
            Global displacement-pressure divergence coupling matrix.
        - ``C`` : sparse matrix
            Global pressure matrix associated with the compressibility term.
        - ``M`` : sparse matrix
            Global displacement mass matrix.
        - ``space`` : dict
            The deterministic degree-of-freedom and element-connectivity maps
            returned by ``build_mini_space``.
    """
    return system
```

### Step 5

05_pressure_condensation.py

Goal
----
Eliminate the pressure variable from the mixed system while retaining the continuous P1 zero-mean pressure constraint.

```python
import numpy as np


def pressure_condense(
    B,
    C,
    pressure_mass,
):
    """
    Prepare the zero-mean pressure Schur complement.

    Parameters
    ----------
    B : scipy.sparse.spmatrix
        Global divergence matrix.
    C : scipy.sparse.spmatrix
        Global pressure bilinear-form matrix.
    pressure_mass : scipy.sparse.spmatrix
        Continuous P1 pressure mass matrix.

    Returns
    -------
    data : dict
        Reduced pressure operators for the zero-mean pressure subspace,
        containing the following required keys:

        - ``Z`` : scipy.sparse.spmatrix
            Zero-mean pressure transformation matrix mapping reduced
            pressure coordinates to the full pressure space.
        - ``Br`` : scipy.sparse.spmatrix
            Reduced displacement-pressure coupling matrix, defined by
            ``Br = Z.T @ B``.
        - ``Cr`` : scipy.sparse.spmatrix
            Reduced pressure matrix, defined by
            ``Cr = Z.T @ C @ Z``.

    Raises
    ------
    ValueError
        If the pressure mean constraint is invalid or if the pressure
        matrices have incompatible shapes.
    """
    return data
```

### Step 6

06_solve_positive_spectrum.py

Goal
----
Compute the lowest positive eigenvalues of the condensed Mini mixed finite-element problem.

```python
import numpy as np


def solve_positive_spectrum(
    A,
    B,
    C,
    M,
    pressure_mass,
    k: int = 4,
    sigma: float = 50.0,
    condensed_pressure=None,
) -> np.ndarray:
    """
    Compute the k smallest positive eigenvalues of the constrained
    mixed displacement-pressure eigenproblem.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Global displacement stiffness matrix.
    B : scipy.sparse.spmatrix
        Global divergence coupling matrix.
    C : scipy.sparse.spmatrix
        Global pressure bilinear-form matrix from the mixed formulation.
    M : scipy.sparse.spmatrix
        Global displacement L2 mass matrix.
    pressure_mass : scipy.sparse.spmatrix
        Continuous P1 pressure mass matrix.
    k : int
        Number of smallest positive eigenvalues to return.
    sigma : float
        Spectral shift used by the eigensolver.
    condensed_pressure : dict[str, scipy.sparse.spmatrix] or None
        Optional zero-mean pressure reduction produced by the pressure
        condensation step. When provided, it contains exactly the entries

            Z  : zero-mean pressure transformation matrix, built by
                 eliminating the last pressure degree of freedom, so its
                 leading n_q - 1 rows are the identity,
            Br : reduced divergence coupling, Br = Z.T @ B,
            Cr : reduced pressure matrix, Cr = Z.T @ C @ Z.

        The matrices define the pressure operator on the zero-mean pressure
        subspace and may be used to solve the constrained eigenproblem
        without reconstructing the reduction.

    Returns
    -------
    eigenvalues : np.ndarray
        Increasing array containing the k smallest positive discrete
        eigenvalues.

    Raises
    ------
    ValueError
        If the matrix dimensions are incompatible, if k is not positive,
        or if sigma is not finite.
    """
    return eigenvalues
```

### Step 7

07_extract_second_eigenvalue.py

Goal
----
Extract the second-smallest positive eigenvalue from an ordered discrete spectrum.

```python
import numpy as np

def extract_second_eigenvalue(eigenvalues: np.ndarray) -> float:
    """
    Extract the second-smallest positive eigenvalue.

    Parameters
    ----------
    eigenvalues : np.ndarray
        One-dimensional array of eigenvalues.

    Returns
    -------
    result : float
        Second-smallest positive eigenvalue.
    """
    return result
```

### Step 8

08_mini_elasticity_orchestrator.py

Goal
----
End-to-end deterministic computation of the requested Mini mixed finite element eigenvalue on the benchmark-specific mesh and material parameters.

```python
import numpy as np
def solve_mini_elasticity(
    nx: int = 48,
    ny: int = 48,
    refine: int = 1,
    mu: float = 1.0,
    lam: float = 7777.0,
) -> float:
    """
    Compute the benchmark Mini mixed finite-element eigenvalue.

    Parameters
    ----------
    nx : int
        Initial number of Cartesian cells in x.
    ny : int
        Initial number of Cartesian cells in y.
    refine : int
        Number of uniform refinements.
    mu : float
        Shear modulus.
    lam : float
        Lame parameter.

    Returns
    -------
    result : float
        Second-smallest positive discrete eigenvalue.
    """
    return result
```
