# Material_Science-Semiconductor_Materials-31

## Background

Carrier transport in semiconductor materials reflects the competition between diffusion and electric drift. Spatially varying generation and contact conditions can produce steep concentration profiles, especially when drift dominates diffusion. Numerical device models must resolve these profiles while respecting the local balance of carriers.

Finite-volume discretizations express transport through fluxes across control-volume boundaries. Higher-order approximations can improve resolution on coarse or nonuniform grids, but their accuracy depends on how multidimensional transport enters the flux reconstruction. Separating normal transport from transverse variations is useful only when the contributions required by the governing conservation law remain coupled.

## Problem

Consider a dimensionless steady carrier-transport problem on $\Omega=[0,1]^2$ with a prescribed constant electric drift, governed by $-\nabla\cdot\mathbf j=f$, $\mathbf j=\alpha\nabla u-\boldsymbol\beta u$, $\alpha=0.03$, $\boldsymbol\beta=(1.7,-0.9)$, $f(x,y)=1+x+2y+3xy$, and Dirichlet data $u=1+0.2x+0.3y$ on the entire boundary.
Use the continuous tensor-product quadratic nodal space on the primary grid $x=(0,0.22,0.57,1)$ and $y=(0,0.31,0.64,1)$, whose local nodes are the endpoints and midpoint of each coordinate interval.
Within each primary rectangle, partition each coordinate at fractions $p=0.27$ and $1-p$ and assign the resulting nine subrectangles to the corresponding nine local nodes, joining pieces that belong to the same global node.
Determine the nodal field from conservative balances over all interior nodal control volumes using the high-order exponential flux reconstruction obtained from the exact local normal boundary-value equation with effective source $f+\partial_\eta j_t$, retaining both the diffusive and advective parts of $j_t=\alpha\partial_\eta u-\beta_tu$.
For every segment of a control-volume face inside a primary element, use its midpoint as the origin, the positive coordinate direction as normal, a perpendicular unit tangent completing a right-handed frame, and symmetric normal trace locations $\xi_K=-0.18h_n$, $\xi_L=0.18h_n$, where $h_n$ is that element's primary width in the normal direction; all traces and derivatives use that element's quadratic polynomial, and all flux and source integrals mean their exact mathematical values.
Impose the boundary nodal values strongly, use each interior face once with opposite signs in its two adjacent balances, and compute the solution without positivity clipping or further stabilization.
Report only the dimensionless integrated carrier population $M=\int_\Omega u_h(x,y)\,dx\,dy$, with absolute error at most $10^{-8}$.
In the reasoning, justify the normal-flux identity, the retained transverse and source contributions, and the role of the freely prescribed symmetric trace distances in light of the method’s reported accuracy evidence, and give the free-node count, $M$, and the maximum absolute interior control-volume balance residual.

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

01_build_dual_partition

Goal
----
Construct quadratic nodal coordinates and their one-dimensional control intervals.

```python
import numpy as np


def build_dual_partition(knots: np.ndarray, partition: float) -> np.ndarray:
    r"""Construct quadratic nodal coordinates and their one-dimensional control intervals.

    Parameters
    ----------
    knots : np.ndarray
        Strictly increasing finite vector of at least two primary coordinates.
    partition : float
        Fraction $p$ satisfying $0 < p < 1/2$.

    Returns
    -------
    np.ndarray
        Shape $(2m-1,3)$ for $m$ knots; coordinate, left cut, right cut.

    Raises
    ------
    ValueError
        If knots are not a finite increasing vector or partition is outside $(0,1/2)$.
    """
    return None
```

### Step 2

02_compute_normal_kernel

Goal
----
Compute the exponentially fitted trace weights and three signed source-kernel moments.

```python
import numpy as np


def compute_normal_kernel(
    half_width: float, diffusion: float, normal_drift: float
) -> np.ndarray:
    r"""Compute the exponentially fitted trace weights and three signed source-kernel moments.

    Parameters
    ----------
    half_width : float
        Positive dimensionless normal half-width $\ell$.
    diffusion : float
        Positive dimensionless diffusion coefficient $\alpha$.
    normal_drift : float
        Finite signed normal drift $\beta_n$.

    Returns
    -------
    np.ndarray
        Shape $(5,)$, containing two trace weights and three moments.

    Raises
    ------
    ValueError
        If scalar data are non-finite, widths or diffusion are non-positive, or $|z|>1000$,
        or if the kernel moments are not representable as finite floats.
    """
    return None
```

### Step 3

03_transform_quadratic_traces

Goal
----
Express the nine quadratic nodal basis functions in oriented face coordinates.

```python
import numpy as np


def transform_quadratic_traces(
    bounds: np.ndarray, center: np.ndarray, axis: int
) -> np.ndarray:
    r"""Express the nine quadratic nodal basis functions in oriented face coordinates.

    Parameters
    ----------
    bounds : np.ndarray
        Shape $(4,)$, ordered as $[x_0,x_1,y_0,y_1]$ with positive widths.
    center : np.ndarray
        Shape $(2,)$; finite face midpoint within the closed rectangle.
    axis : int
        Zero for a positive horizontal normal, one for a positive vertical normal.

    Returns
    -------
    np.ndarray
        Shape $(9,3,3)$; local node, ascending normal power, ascending tangent power.

    Raises
    ------
    ValueError
        If shapes, finite values, rectangle widths, midpoint location, or axis are invalid.
    """
    return None
```

### Step 4

04_compute_homogeneous_face

Goal
----
Evaluate the homogeneous integrated normal-flux functional of a quadratic element.

```python
import numpy as np


def compute_homogeneous_face(
    coefficients: np.ndarray,
    half_width: float,
    face_length: float,
    diffusion: float,
    kernel: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the homogeneous integrated normal-flux functional of a quadratic element.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape $(9,3,3)$; basis coefficients in powers of normal and tangent coordinates.
    half_width : float
        Positive normal half-width $\ell$.
    face_length : float
        Positive tangent length $H$.
    diffusion : float
        Positive diffusion coefficient $\alpha$.
    kernel : np.ndarray
        Shape $(5,)$, $[B(z),B(-z),I_0,I_1,I_2]$ for the same normal interval.

    Returns
    -------
    np.ndarray
        Shape $(9,)$; coefficients of the positive-normal homogeneous flux.

    Raises
    ------
    ValueError
        If any shape or finite-value requirement fails, a length or diffusion is non-positive, or either trace weight is negative.
    """
    return None
```

### Step 5

05_compute_transverse_face

Goal
----
Evaluate the signed transverse contribution to the integrated normal flux.

```python
import numpy as np


def compute_transverse_face(
    coefficients: np.ndarray,
    face_length: float,
    diffusion: float,
    tangent_drift: float,
    moments: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the signed transverse contribution to the integrated normal flux.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape $(9,3,3)$; local basis polynomial tensor.
    face_length : float
        Positive face length $H$.
    diffusion : float
        Positive diffusion coefficient $\alpha$.
    tangent_drift : float
        Finite signed tangent drift $\beta_t$ in the oriented frame.
    moments : np.ndarray
        Shape $(3,)$, signed moments $I_0,I_1,I_2$.

    Returns
    -------
    np.ndarray
        Shape $(9,)$; transverse contribution to the positive-normal flux.

    Raises
    ------
    ValueError
        If shapes or finite-value requirements fail, or face length or diffusion is non-positive.
    """
    return None
```

### Step 6

06_compute_source_face

Goal
----
Integrate the prescribed generation term against the signed normal-flux kernel.

```python
import numpy as np


def compute_source_face(
    source: np.ndarray,
    center: np.ndarray,
    axis: int,
    face_length: float,
    moments: np.ndarray,
) -> float:
    r"""Integrate the prescribed generation term against the signed normal-flux kernel.

    Parameters
    ----------
    source : np.ndarray
        Shape $(4,)$, ordered as $[f_0,f_x,f_y,f_{xy}]$.
    center : np.ndarray
        Shape $(2,)$; finite face midpoint.
    axis : int
        Zero for a positive horizontal normal or one for a positive vertical normal.
    face_length : float
        Positive tangent length $H$.
    moments : np.ndarray
        Shape $(3,)$, signed moments $I_0,I_1,I_2$.

    Returns
    -------
    float
        Integrated positive-normal source flux $J^f$.

    Raises
    ------
    ValueError
        If shapes, finite values or axis are invalid, or face length is non-positive.
    """
    return None
```

### Step 7

07_solve_carrier_balance

Goal
----
Assemble and solve the conservative quadratic carrier-balance equations.

```python
import numpy as np


def solve_carrier_balance(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> np.ndarray:
    r"""Assemble and solve the conservative quadratic carrier-balance equations.

    Parameters
    ----------
    x_knots, y_knots : np.ndarray
        Strictly increasing finite primary coordinate vectors, each of length at least two.
    diffusion : float
        Positive dimensionless diffusion coefficient.
    drift : np.ndarray
        Shape $(2,)$; finite constant horizontal and vertical drift.
    source : np.ndarray
        Shape $(4,)$; coefficients $[f_0,f_x,f_y,f_{xy}]$.
    boundary : np.ndarray
        Shape $(3,)$; Dirichlet coefficients $[g_0,g_x,g_y]$.
    partition : float, optional
        Dual cut fraction $p$ in $(0,1/2)$.
    trace_fraction : float, optional
        Positive half-width fraction $r<p$; all face Peclet magnitudes must be at most 1000.

    Returns
    -------
    np.ndarray
        Shape $(2m_y-1,2m_x-1)$, the nodal carrier field including contacts.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements fail, knots are not increasing,
        diffusion is non-positive, $0<r<p<1/2$ fails, a face Peclet magnitude
        exceeds 1000, or the discrete system is singular or has no finite solution.
    """
    return None
```

### Step 8

08_compute_carrier_population

Goal
----
Return the integrated carrier population from the complete discrete transport solve.

```python
import numpy as np


def compute_carrier_population(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> float:
    r"""Return the integrated carrier population from the complete discrete transport solve.

    Parameters
    ----------
    x_knots, y_knots : np.ndarray
        Strictly increasing finite primary coordinate vectors, each of length at least two.
    diffusion : float
        Positive dimensionless diffusion coefficient.
    drift : np.ndarray
        Shape $(2,)$; finite constant horizontal and vertical drift.
    source : np.ndarray
        Shape $(4,)$; coefficients $[f_0,f_x,f_y,f_{xy}]$.
    boundary : np.ndarray
        Shape $(3,)$; Dirichlet coefficients $[g_0,g_x,g_y]$.
    partition : float, optional
        Dual cut fraction $p$ in $(0,1/2)$.
    trace_fraction : float, optional
        Positive half-width fraction $r<p$; all face Peclet magnitudes must be at most 1000.

    Returns
    -------
    float
        Dimensionless integrated carrier population over the full rectangular domain.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements fail, knots are not increasing,
        diffusion is non-positive, $0<r<p<1/2$ fails, a face Peclet magnitude
        exceeds 1000, or the discrete system is singular or has no finite solution,
        or the integrated population is not finite.
    """
    return None
```
