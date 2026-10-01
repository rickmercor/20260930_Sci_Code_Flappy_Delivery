# Mathematics-Numerical_Linear_Algebra-62

## Background

A Schur-based mixed-precision square-root method can recover working-precision accuracy from a lower-precision matrix-function approximation by evaluating residuals in higher precision and reusing structured correction information. A higher-order matrix root-finding update provides a distinct local mechanism: it modifies the Newton correction using second-derivative information, and its cubic behavior is characterized by a matrix Schwarzian quantity. Comparing both corrections from the same lower-precision starting point isolates the effect of the higher-order update without changing the initial approximation.

## Problem

Consider the real matrix

$$
A=\begin{bmatrix}
734.3125 & -562.1875 & -729.6875 & 557.8125\\
320.3125 & -448.1875 & -319.6875 & 447.8125\\
-809.6875 & 917.8125 & 814.3125 & -922.1875\\
-199.6875 & 7.8125 & 200.3125 & -8.1875
\end{bmatrix},
$$

with supplied lower-precision Schur data

$$
Q=\begin{bmatrix}
0.6966455578804016 & 0.4486113488674164 & 0.5052264332771301 & -0.24120382964611053\\
0.21158075332641602 & -0.4446275234222412 & 0.4567680060863495 & 0.7408797144889832\\
-0.6356939673423767 & 0.5600026845932007 & 0.48396551609039307 & 0.21924364566802979\\
-0.2565384805202484 & -0.5361445546150208 & 0.5494421124458313 & -0.5872394442558289
\end{bmatrix},
$$

$$
T=\begin{bmatrix}
1024.0001220703125 & -225.39634704589844 & 123.67469787597656 & -1950.7784423828125\\
0 & 64.0001449584961 & -45.988956451416016 & 683.4092407226562\\
0 & 0 & 3.9994139671325684 & -56.81679916381836\\
0 & 0 & 0 & 0.25045859813690186
\end{bmatrix}.
$$

Convert the supplied entries of $Q$ and $T$ once to IEEE-754 binary32; do not recompute a Schur factorization. Let $S$ be the principal upper-triangular square root of $T$ and $X_0=QSQ^T$ the binary32 initial approximation. Form the working-precision residual $R_0=A-X_0^2$ in binary64. For the structured mixed-precision path, transform $R_0$ as $\widehat R_0=\mathrm{fl}_{32}(Q_{64}^TR_0Q_{64})$, solve the frozen Schur-coordinate Sylvester correction with $\mathrm{blks}=1$, lift the correction in binary32, update in binary64, and denote the resulting residual norm by $r_{\mathrm{MP}}$.

From the same $X_0$, independently consider the nonlinear matrix equation

$$
F(X)=X^2-A=0.
$$

Its first and second Fréchet derivatives are

$$
\mathcal A_X[V]=XV+VX,
\qquad
\mathcal B[U,V]=UV+VU.
$$

Let the Newton direction $H$ be the unique solution of

$$
\mathcal A_{X_0}[H]=R_0,
$$

and let the matrix Halley correction $Y$ be the unique solution of

$$
\left(\mathcal A_{X_0}+\frac12\mathcal B[H,\cdot]\right)[Y]=R_0.
$$

Set $X_H=X_0+Y$ and $r_H=\|A-X_H^2\|_F$. Because $D^3F=0$ for this quadratic matrix map, define the directional matrix Schwarzian contraction at $X_0$ by

$$
\mathcal S_{X_0}[H,H,H]
=-3\,\mathcal A_{X_0}^{-1}\!\left(
\mathcal B\!\left[H,\mathcal A_{X_0}^{-1}\mathcal B[H,H]\right]
\right).
$$

The same third-order construction has a polarized form. For arbitrary directions $U,V,W$, define

$$
\begin{aligned}
\mathcal S_{X_0}[U,V,W] ={}&-\mathcal A_{X_0}^{-1}\!\left(\mathcal B\!\left[U,\mathcal A_{X_0}^{-1}\mathcal B[V,W]\right]\right)\\
&-\mathcal A_{X_0}^{-1}\!\left(\mathcal B\!\left[V,\mathcal A_{X_0}^{-1}\mathcal B[U,W]\right]\right)\\
&-\mathcal A_{X_0}^{-1}\!\left(\mathcal B\!\left[W,\mathcal A_{X_0}^{-1}\mathcal B[U,V]\right]\right).
\end{aligned}
$$

For the benchmark also evaluate the genuinely mixed-direction diagnostic with $U=H$, $V=R_0$, and $W=I$, and let

$$
p_H=\|\mathcal S_{X_0}[H,R_0,I]\|_F.
$$

Let $s_H=\|\mathcal S_{X_0}[H,H,H]\|_F$ and define the Halley advantage

$$
\Gamma=\frac{r_{\mathrm{MP}}}{r_H}.
$$

Every binary32 scalar operation rounds immediately to binary32, every binary64 scalar operation rounds immediately to binary64, matrix dot products accumulate from positive zero in increasing inner-index order, Frobenius norms accumulate squared entries in row-major order in binary64, and triple products are evaluated left to right with the intermediate stored in the stated precision. For deterministic dense Fréchet inverse applications, explicit operator matrices act on column-stacked $\operatorname{vec}(V)$ (columns concatenated in increasing column index). For the Halley correction, form the linear operator directly as $\mathcal A_M$ at $M=X_0+\tfrac12H$ in binary64. Solve these dense systems by binary64 elimination with the first maximum-magnitude pivot in increasing row order; do not replace the specified scalar path by a higher-level matrix inverse, Kronecker solve, eigendecomposition, or SVD. Whenever a Sylvester right-hand side is reduced by two block products, subtract the product with the coefficient acting from the left before the product with the coefficient acting from the right.

Report $\Gamma$ to at least six significant digits. In `<reasoning>`, report $s_{11}$, $\eta_0=\|R_0\|_F/\|A\|_F$, $\|\widehat R_0\|_F$, $r_{\mathrm{MP}}$, $H_{42}$, $Y_{42}$, $r_H$, $s_H$, $p_H$, and the unrounded $\Gamma$.

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

step_01_triangular_principal_sqrt

Goal
----
Compute the deterministic binary32 principal square root of an upper-triangular matrix.

```python
import numpy as np

def triangular_principal_sqrt(T: np.ndarray) -> np.ndarray:
    """Compute the deterministic binary32 principal square root of an upper-triangular matrix.

    Parameters
    ----------
    T : np.ndarray
        Finite nonempty upper-triangular matrix with strictly positive diagonal.

    Returns
    -------
    S : np.ndarray
        Binary32 upper-triangular principal square root under the task-wide deterministic arithmetic convention.

    Raises
    ------
    ValueError
        If `T` is empty, non-square, non-finite, non-triangular, or has a non-positive diagonal."""
    return S
```

### Step 2

step_02_reconstruct_initial_sqrt

Goal
----
Reconstruct the lower-precision initial square-root approximation.

```python
import numpy as np

def reconstruct_initial_sqrt(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Reconstruct the lower-precision initial square-root approximation.

    Parameters
    ----------
    Q : np.ndarray
        Supplied binary32 Schur-vector factor.
    S : np.ndarray
        Compatible binary32 principal square-root factor.

    Returns
    -------
    X0 : np.ndarray
        Binary32 initial square-root approximation in the original coordinates under the task-wide deterministic arithmetic convention.

    Raises
    ------
    ValueError
        If the factors are empty, non-square, shape-incompatible, or non-finite."""
    return X0
```

### Step 3

step_03_working_residual_backward_error

Goal
----
Evaluate the square-root residual and normalized backward error in working precision.

```python
import numpy as np

def working_residual_backward_error(A: np.ndarray, X0: np.ndarray) -> tuple[np.ndarray, float]:
    """Evaluate the square-root residual and normalized backward error in working precision.

    Parameters
    ----------
    A : np.ndarray
        Original finite square matrix.
    X0 : np.ndarray
        Compatible initial square-root approximation.

    Returns
    -------
    R0 : np.ndarray
        Binary64 square-root residual.
    eta0 : float
        Binary64 normalized Frobenius backward error.

    Raises
    ------
    ValueError
        If the matrices are empty, non-square, shape-incompatible, non-finite, or `A` has zero Frobenius norm."""
    return R0, eta0
```

### Step 4

step_04_schur_residual_transform

Goal
----
Transform the working residual to the supplied Schur coordinates.

```python
import numpy as np

def schur_residual_transform(Q: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform the working residual to the supplied Schur coordinates.

    Parameters
    ----------
    Q : np.ndarray
        Supplied Schur-vector factor.
    R0 : np.ndarray
        Compatible binary64 working residual.

    Returns
    -------
    Rhat : np.ndarray
        Binary32 Schur-coordinate residual under the task-wide deterministic mixed-precision convention.
    rhat_norm : float
        Binary64 Frobenius norm of `Rhat`.

    Raises
    ------
    ValueError
        If the inputs are empty, non-square, shape-incompatible, or non-finite."""
    return Rhat, rhat_norm
```

### Step 5

step_05_block_recursive_sylvester

Goal
----
Solve the triangular Fréchet correction by real-Schur block recursion.

```python
import numpy as np

def block_recursive_sylvester(
    S: np.ndarray,
    Stilde: np.ndarray,
    R: np.ndarray,
    blks: int,
    return_profile: bool = False,
    power: int = 2,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int, int]]:
    """Solve the triangular Fréchet correction by real-Schur block recursion.

    Parameters
    ----------
    S : np.ndarray
        Finite square real Schur-form left coefficient with nonoverlapping 1x1 and 2x2 diagonal blocks.
    Stilde : np.ndarray
        Finite square real Schur-form right coefficient with the same structural convention.
    R : np.ndarray
        Finite compatible right-hand side.
    blks : int
        Positive minimal block size for the recursive solver. The direct solver is
        used only when both dimensions are at most `blks`. Otherwise a side is
        partitioned at its midpoint, except that a boundary which would cut a 2x2
        real-Schur diagonal block moves to the nearest legal boundary, with an exact
        distance tie resolved to the lower index. If the side selected by the
        recursion has no legal boundary, the other side is partitioned instead; if
        neither side has one, that subproblem is solved directly.
    power : int, default=2
        Root power selecting the paper-specific Fréchet correction; supported values are 2 and 3.
    return_profile : bool, optional
        Whether to return recursion-profile diagnostics with the correction.

    Returns
    -------
    Y : np.ndarray
        Binary32 solution of the selected triangular Fréchet correction under the prescribed real-Schur block-recursive method and task-wide deterministic arithmetic convention.
    profile : tuple, optional
        Returned only when `return_profile=True`. Five integers giving, in this
        order, the number of row splits, the number of column splits, the number of
        combined row-and-column splits, the number of direct solves performed, and
        the maximum recursion depth reached.

    Raises
    ------
    ValueError
        If an argument violates the stated domain or the required direct Fréchet subproblem is singular."""
    return Y
```

### Step 6

step_06_lift_validate_correction

Goal
----
Lift the structured correction and evaluate refinement diagnostics.

```python
import numpy as np

def lift_validate_correction(
    Q: np.ndarray,
    X0: np.ndarray,
    E: np.ndarray,
    R0: np.ndarray,
    return_certificate: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float, bool]:
    """Lift the structured correction and evaluate refinement diagnostics.

    Parameters
    ----------
    Q : np.ndarray
        Supplied Schur-vector factor.
    X0 : np.ndarray
        Initial square-root approximation.
    E : np.ndarray
        Schur-coordinate correction.
    R0 : np.ndarray
        Working-precision residual.
    return_certificate : bool, default=False
        Whether to include the local frozen-Fréchet convergence certificate.

    Returns
    -------
    DeltaX : np.ndarray
        Binary32 correction in the original coordinates.
    rho : float
        Binary64 normalized defect of the lifted correction.
    theta : float, optional
        Local convergence-certificate value for the frozen Fréchet map, returned only when requested.
    condition_holds : bool, optional
        Whether the certificate satisfies its prescribed convergence threshold.

    Raises
    ------
    ValueError
        If an input violates the matrix domain, `return_certificate` is not boolean, or a requested certificate is undefined."""
    return DeltaX, rho
```

### Step 7

step_07_working_precision_update

Goal
----
Apply a correction in binary64 and evaluate the next residual.

```python
import numpy as np

def working_precision_update(A: np.ndarray, Xk: np.ndarray, DeltaXk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a correction in binary64 and evaluate the next residual.

    Parameters
    ----------
    A : np.ndarray
        Original finite square matrix.
    Xk : np.ndarray
        Current square-root approximation.
    DeltaXk : np.ndarray
        Compatible correction.

    Returns
    -------
    Xnext : np.ndarray
        Binary64 updated approximation.
    Rnext : np.ndarray
        Binary64 residual of the updated approximation.
    residual_norm : float
        Binary64 Frobenius norm of `Rnext`.

    Raises
    ------
    ValueError
        If inputs are empty, non-square, shape-incompatible, or non-finite."""
    return Xnext, Rnext, residual_norm
```

### Step 8

step_08_halley_newton_direction

Goal
----
Return the binary64 Newton direction for a matrix-power residual.

```python
import numpy as np

def halley_newton_direction(A: np.ndarray, X: np.ndarray, power: int = 2) -> np.ndarray:
    """Return the binary64 Newton direction for a matrix-power residual.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square target matrix.
    X : np.ndarray
        Finite compatible iterate.
    power : int, default=2
        Supported residual-map power, 2 or 3.

    Returns
    -------
    H : np.ndarray
        Binary64 Newton direction obtained from the Fréchet derivative of the selected noncommutative matrix-power map.

    Raises
    ------
    ValueError
        If the inputs violate the matrix domain, `power` is unsupported, or the required Fréchet derivative is singular."""
    return H
```

### Step 9

step_09_matrix_halley_schwarzian

Goal
----
Return a matrix-Halley correction and matrix-Schwarzian diagnostics.

```python
import numpy as np

def matrix_halley_schwarzian(
    A: np.ndarray,
    X: np.ndarray,
    H: np.ndarray,
    directions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    power: int = 2,
) -> tuple[np.ndarray, float, float, float]:
    """Return a matrix-Halley correction and matrix-Schwarzian diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square target matrix.
    X : np.ndarray
        Finite compatible iterate.
    H : np.ndarray
        Compatible Newton direction.
    directions : tuple, optional
        Optional ordered triple of compatible directions for the polarized Schwarzian; otherwise use the benchmark directions.
    power : int, default=2
        Supported residual-map power, 2 or 3.

    Returns
    -------
    Y : np.ndarray
        Binary64 matrix-Halley correction.
    residual_norm : float
        Binary64 residual norm after the Halley update.
    schwarzian_norm : float
        Frobenius norm of the repeated-direction matrix Schwarzian.
    polarized_norm : float
        Frobenius norm of the selected polarized matrix Schwarzian contraction.

    Raises
    ------
    ValueError
        If the inputs violate the stated domain, `power` is unsupported, a required Fréchet operator is singular, or a diagnostic is non-finite."""
    return Y, halley_residual, directional_norm, polarized_norm
```

### Step 10

step_10_integrated_halley_advantage

Goal
----
Run the complete pipeline and return the final comparison diagnostics.

```python
import numpy as np

def integrated_halley_advantage(A: np.ndarray, T: np.ndarray, Q: np.ndarray) -> tuple[float,float,float,float]:
    """Run the complete pipeline and return the final comparison diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Original finite nonempty square matrix.
    T : np.ndarray
        Supplied compatible binary32 upper-triangular Schur factor.
    Q : np.ndarray
        Supplied compatible binary32 Schur-vector factor.

    Returns
    -------
    advantage : float
        Ratio of the mixed-precision refinement residual to the matrix-Halley residual.
    schwarzian_norm : float
        Repeated-direction matrix-Schwarzian norm.
    polarized_norm : float
        Mixed-direction polarized matrix-Schwarzian norm.
    halley_residual : float
        Residual norm after the Halley update.

    Raises
    ------
    ValueError
        If supplied data violate a required domain condition, a required Fréchet operator is singular, or the final ratio is undefined."""
    return advantage, schwarzian_norm, polarized_norm, halley_residual
```
